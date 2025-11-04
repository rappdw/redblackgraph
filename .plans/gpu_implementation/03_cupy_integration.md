# CuPy Integration and Python API

## Overview

This document describes how to integrate CUDA kernels with CuPy and design the Python API for GPU-accelerated red-black graph operations.

## CuPy RawKernel Approach

CuPy's `RawKernel` allows us to write CUDA C++ code and call it from Python without a full build system.

### Benefits
- **Rapid prototyping**: No separate compilation step
- **JIT compilation**: Kernels compiled on first use
- **Caching**: Compiled kernels cached between runs
- **Templating**: Easy to handle multiple dtypes
- **Debugging**: Can print from kernels during development

### Example: AVOS Sum Kernel

```python
import cupy as cp

avos_sum_kernel_src = r'''
template<typename T>
__device__ T avos_sum(T x, T y) {
    if (x == 0) return y;
    if (y == 0) return x;
    using UT = typename std::make_unsigned<T>::type;
    return (static_cast<UT>(x) < static_cast<UT>(y)) ? x : y;
}

extern "C" __global__
void avos_sum_kernel(const long long* x, const long long* y, 
                     long long* out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = avos_sum(x[idx], y[idx]);
    }
}
'''

avos_sum_kernel = cp.RawKernel(avos_sum_kernel_src, 'avos_sum_kernel')
```

## Kernel Management Class

### Design: Lazy-Loading Kernel Manager

```python
# redblackgraph/gpu/kernels/__init__.py
import cupy as cp
from pathlib import Path

class AVOSKernels:
    """Manages CUDA kernels for AVOS operations.
    
    Kernels are loaded lazily on first use and cached.
    Supports multiple dtypes via C++ templates.
    """
    
    def __init__(self):
        self._kernels = {}
        self._kernel_dir = Path(__file__).parent
        
    def _load_kernel_source(self, filename):
        """Load CUDA source from file."""
        path = self._kernel_dir / filename
        return path.read_text()
    
    def _get_kernel(self, name, dtype):
        """Get or compile kernel for specific dtype."""
        key = (name, dtype)
        if key not in self._kernels:
            self._kernels[key] = self._compile_kernel(name, dtype)
        return self._kernels[key]
    
    def _compile_kernel(self, name, dtype):
        """Compile kernel with appropriate template instantiation."""
        # Load kernel source
        if name in ['avos_sum', 'avos_product']:
            source = self._load_kernel_source('avos_kernels.cu')
        elif name.startswith('sparse_'):
            source = self._load_kernel_source('sparse_kernels.cu')
        else:
            raise ValueError(f"Unknown kernel: {name}")
        
        # Map numpy dtype to C++ type
        dtype_map = {
            cp.int8: 'char',
            cp.int16: 'short',
            cp.int32: 'int',
            cp.int64: 'long long',
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Template instantiation handled by RawKernel
        return cp.RawKernel(source, f'{name}_kernel',
                           options=('--std=c++14',))
    
    def avos_sum(self, x, y, out=None):
        """Element-wise AVOS sum.
        
        Args:
            x: CuPy array
            y: CuPy array (same shape and dtype as x)
            out: Output array (optional, created if None)
            
        Returns:
            CuPy array containing avos_sum(x, y)
        """
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        if x.dtype != y.dtype:
            raise ValueError(f"Dtype mismatch: {x.dtype} != {y.dtype}")
        
        if out is None:
            out = cp.empty_like(x)
        
        kernel = self._get_kernel('avos_sum', x.dtype)
        
        threads_per_block = 256
        blocks = (x.size + threads_per_block - 1) // threads_per_block
        
        kernel((blocks,), (threads_per_block,),
               (x, y, out, x.size))
        
        return out
    
    def avos_product(self, x, y, out=None):
        """Element-wise AVOS product with parity constraints.
        
        Args:
            x: CuPy array (left operand)
            y: CuPy array (right operand, same shape/dtype as x)
            out: Output array (optional)
            
        Returns:
            CuPy array containing avos_product(x, y)
        """
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        if x.dtype != y.dtype:
            raise ValueError(f"Dtype mismatch: {x.dtype} != {y.dtype}")
        
        if out is None:
            out = cp.empty_like(x)
        
        kernel = self._get_kernel('avos_product', x.dtype)
        
        threads_per_block = 256
        blocks = (x.size + threads_per_block - 1) // threads_per_block
        
        kernel((blocks,), (threads_per_block,),
               (x, y, out, x.size))
        
        return out

# Global instance
_kernels = AVOSKernels()

def avos_sum(x, y, out=None):
    """Global function for AVOS sum."""
    return _kernels.avos_sum(x, y, out)

def avos_product(x, y, out=None):
    """Global function for AVOS product."""
    return _kernels.avos_product(x, y, out)
```

## GPU Matrix Class

### API Design: rb_matrix_gpu

```python
# redblackgraph/gpu/rb_matrix_gpu.py
import cupy as cp
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from ..sparse import rb_matrix
from .kernels import avos_sum, avos_product

class rb_matrix_gpu:
    """GPU-accelerated sparse Red-Black matrix using AVOS semiring.
    
    Stores matrix in CSR (Compressed Sparse Row) format on GPU.
    Compatible with rb_matrix API for drop-in replacement.
    
    Attributes:
        data (cp.ndarray): Non-zero values
        indices (cp.ndarray): Column indices (int32)
        indptr (cp.ndarray): Row pointers (int32)
        shape (tuple): Matrix dimensions (m, n)
        dtype: NumPy dtype of elements
        device (int): GPU device ID
    
    Examples:
        >>> # Create from scipy sparse matrix
        >>> A_cpu = rb_matrix(scipy_csr)
        >>> A_gpu = rb_matrix_gpu(A_cpu)
        >>> 
        >>> # Matrix multiplication on GPU
        >>> C_gpu = A_gpu @ A_gpu
        >>> 
        >>> # Transfer back to CPU
        >>> C_cpu = C_gpu.to_cpu()
        >>> 
        >>> # Batch operations on GPU (efficient)
        >>> A2 = A_gpu @ A_gpu
        >>> A3 = A2 @ A_gpu
        >>> A4 = A3 @ A_gpu
        >>> result = A4.to_cpu()
    """
    
    def __init__(self, matrix, device=0):
        """Create GPU matrix from CPU matrix or arrays.
        
        Args:
            matrix: rb_matrix, scipy csr_matrix, numpy array, 
                    or tuple (data, indices, indptr, shape, dtype)
            device: GPU device ID (default 0)
        """
        self.device = device
        
        with cp.cuda.Device(device):
            if isinstance(matrix, rb_matrix_gpu):
                # Copy from another GPU matrix
                self.data = cp.copy(matrix.data)
                self.indices = cp.copy(matrix.indices)
                self.indptr = cp.copy(matrix.indptr)
                self.shape = matrix.shape
                self.dtype = matrix.dtype
                
            elif isinstance(matrix, (rb_matrix, csr_matrix)) or isspmatrix_csr(matrix):
                # Transfer from CPU sparse matrix
                cpu_mat = matrix.tocsr() if not isspmatrix_csr(matrix) else matrix
                
                self.data = cp.asarray(cpu_mat.data)
                self.indices = cp.asarray(cpu_mat.indices, dtype=cp.int32)
                self.indptr = cp.asarray(cpu_mat.indptr, dtype=cp.int32)
                self.shape = cpu_mat.shape
                self.dtype = cpu_mat.dtype
                
            elif isinstance(matrix, tuple):
                # Direct construction from arrays
                data, indices, indptr, shape, dtype = matrix
                self.data = cp.asarray(data, dtype=dtype)
                self.indices = cp.asarray(indices, dtype=cp.int32)
                self.indptr = cp.asarray(indptr, dtype=cp.int32)
                self.shape = shape
                self.dtype = dtype
                
            elif isinstance(matrix, np.ndarray):
                # Convert dense array to sparse
                cpu_sparse = csr_matrix(matrix)
                self.data = cp.asarray(cpu_sparse.data)
                self.indices = cp.asarray(cpu_sparse.indices, dtype=cp.int32)
                self.indptr = cp.asarray(cpu_sparse.indptr, dtype=cp.int32)
                self.shape = matrix.shape
                self.dtype = matrix.dtype
                
            else:
                raise TypeError(f"Cannot create rb_matrix_gpu from {type(matrix)}")
    
    def __matmul__(self, other):
        """Matrix multiplication using AVOS semiring.
        
        Args:
            other: rb_matrix_gpu with compatible shape
            
        Returns:
            rb_matrix_gpu containing self @ other
            
        Raises:
            ValueError: If shapes are incompatible
            TypeError: If other is not rb_matrix_gpu
        """
        if not isinstance(other, rb_matrix_gpu):
            raise TypeError(
                f"Cannot multiply rb_matrix_gpu with {type(other)}. "
                f"Convert to GPU first: other_gpu = rb_matrix_gpu(other)"
            )
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Incompatible shapes for matmul: "
                f"{self.shape} @ {other.shape}"
            )
        
        if self.dtype != other.dtype:
            raise ValueError(
                f"Dtype mismatch: {self.dtype} @ {other.dtype}"
            )
        
        with cp.cuda.Device(self.device):
            return self._sparse_matmul(other)
    
    def _sparse_matmul(self, other):
        """Sparse matrix multiplication implementation.
        
        Uses two-pass algorithm:
        1. Count non-zeros per output row
        2. Compute values and column indices
        """
        from .sparse_matmul import sparse_matmul_csr
        
        C_data, C_indices, C_indptr = sparse_matmul_csr(
            self.data, self.indices, self.indptr,
            other.data, other.indices, other.indptr,
            self.shape[0], other.shape[1]
        )
        
        return rb_matrix_gpu(
            (C_data, C_indices, C_indptr, 
             (self.shape[0], other.shape[1]), 
             self.dtype),
            device=self.device
        )
    
    def to_cpu(self):
        """Transfer matrix back to CPU as rb_matrix.
        
        Returns:
            rb_matrix containing the same data
        """
        cpu_data = cp.asnumpy(self.data)
        cpu_indices = cp.asnumpy(self.indices)
        cpu_indptr = cp.asnumpy(self.indptr)
        
        cpu_sparse = csr_matrix(
            (cpu_data, cpu_indices, cpu_indptr),
            shape=self.shape,
            dtype=self.dtype
        )
        
        return rb_matrix(cpu_sparse)
    
    def transitive_closure(self):
        """Compute transitive closure using AVOS semiring.
        
        Returns:
            rb_matrix_gpu containing A* (transitive closure)
            
        Algorithm:
            Warshall's algorithm adapted for AVOS semiring
            A* = A + A² + A³ + ... (until convergence)
        """
        from .transitive_closure import transitive_closure_gpu
        return transitive_closure_gpu(self)
    
    @property
    def nnz(self):
        """Number of stored values (including explicit zeros)."""
        return int(self.data.size)
    
    def __repr__(self):
        return (
            f"<{self.shape[0]}x{self.shape[1]} sparse matrix "
            f"on GPU (device {self.device}) with {self.nnz} stored elements "
            f"in Compressed Sparse Row format, dtype={self.dtype}>"
        )
    
    def __str__(self):
        # Transfer small matrices for display
        if self.nnz < 100:
            return str(self.to_cpu())
        else:
            return repr(self)
```

## Sparse Matrix Multiplication

```python
# redblackgraph/gpu/sparse_matmul.py
import cupy as cp
from .kernels import AVOSKernels

_kernels = AVOSKernels()

def sparse_matmul_csr(A_data, A_indices, A_indptr,
                      B_data, B_indices, B_indptr,
                      m, n):
    """Sparse CSR matrix multiplication using AVOS semiring.
    
    Computes C = A @ B where A is m×k and B is k×n.
    
    Args:
        A_data, A_indices, A_indptr: CSR format for A
        B_data, B_indices, B_indptr: CSR format for B
        m: Number of rows in A (and C)
        n: Number of columns in B (and C)
        
    Returns:
        (C_data, C_indices, C_indptr) in CSR format
    """
    # Pass 1: Count non-zeros per row
    nnz_counts = _count_nnz_per_row(
        A_data, A_indices, A_indptr,
        B_data, B_indices, B_indptr,
        m
    )
    
    # Build indptr via prefix sum
    C_indptr = cp.zeros(m + 1, dtype=cp.int32)
    C_indptr[1:] = cp.cumsum(nnz_counts)
    
    total_nnz = int(C_indptr[-1])
    
    if total_nnz == 0:
        # Result is empty matrix
        return (
            cp.array([], dtype=A_data.dtype),
            cp.array([], dtype=cp.int32),
            C_indptr
        )
    
    # Pass 2: Compute values
    C_data = cp.zeros(total_nnz, dtype=A_data.dtype)
    C_indices = cp.zeros(total_nnz, dtype=cp.int32)
    
    _compute_matmul_values(
        A_data, A_indices, A_indptr,
        B_data, B_indices, B_indptr,
        C_data, C_indices, C_indptr,
        m
    )
    
    return C_data, C_indices, C_indptr

def _count_nnz_per_row(A_data, A_indices, A_indptr,
                       B_data, B_indices, B_indptr, m):
    """Count non-zeros in each row of A @ B."""
    kernel = _kernels._get_kernel('count_nnz', A_data.dtype)
    
    nnz_counts = cp.zeros(m, dtype=cp.int32)
    
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    
    # Hash table size per warp (shared memory)
    HASH_SIZE = 128  # Tune based on typical sparsity
    shared_mem = (threads_per_block // 32) * HASH_SIZE * 4
    
    kernel((blocks,), (threads_per_block,),
           (A_data, A_indices, A_indptr,
            B_data, B_indices, B_indptr,
            nnz_counts, m),
           shared_mem=shared_mem)
    
    return nnz_counts

def _compute_matmul_values(A_data, A_indices, A_indptr,
                          B_data, B_indices, B_indptr,
                          C_data, C_indices, C_indptr, m):
    """Compute values for C = A @ B."""
    kernel = _kernels._get_kernel('compute_values', A_data.dtype)
    
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    
    HASH_SIZE = 128
    shared_mem = (threads_per_block // 32) * HASH_SIZE * 8  # int + value
    
    kernel((blocks,), (threads_per_block,),
           (A_data, A_indices, A_indptr,
            B_data, B_indices, B_indptr,
            C_data, C_indices, C_indptr, m),
           shared_mem=shared_mem)
```

## Memory Management

```python
# redblackgraph/gpu/memory.py
import cupy as cp
from contextlib import contextmanager

class GPUMemoryManager:
    """Manages GPU memory allocation and transfer."""
    
    def __init__(self, device=0):
        self.device = device
        
    @contextmanager
    def on_device(self):
        """Context manager for GPU operations on specific device."""
        with cp.cuda.Device(self.device):
            yield
    
    def get_memory_info(self):
        """Get current GPU memory usage.
        
        Returns:
            dict with 'used', 'total', 'free' in bytes
        """
        mempool = cp.get_default_memory_pool()
        return {
            'used': mempool.used_bytes(),
            'total': mempool.total_bytes(),
            'free': mempool.total_bytes() - mempool.used_bytes()
        }
    
    def free_cache(self):
        """Free unused cached memory."""
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
```

## Public API

```python
# redblackgraph/gpu/__init__.py
"""GPU-accelerated Red-Black Graph operations.

Requires CuPy and CUDA-capable GPU.
"""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False

if CUPY_AVAILABLE and CUDA_AVAILABLE:
    from .rb_matrix_gpu import rb_matrix_gpu
    from .kernels import avos_sum, avos_product
    from .memory import GPUMemoryManager
    
    __all__ = [
        'rb_matrix_gpu',
        'avos_sum',
        'avos_product',
        'GPUMemoryManager',
    ]
else:
    __all__ = []

def __getattr__(name):
    """Provide helpful error messages when GPU not available."""
    if not CUPY_AVAILABLE:
        raise ImportError(
            f"GPU support requires CuPy. Install with:\n"
            f"  pip install cupy-cuda11x  # For CUDA 11.x\n"
            f"  pip install cupy-cuda12x  # For CUDA 12.x\n"
            f"Or see: https://docs.cupy.dev/en/stable/install.html"
        )
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA GPU not available. Check that:\n"
            f"  1. NVIDIA GPU is installed\n"
            f"  2. CUDA drivers are installed\n"
            f"  3. GPU is not in use by another process"
        )
    raise AttributeError(f"module 'redblackgraph.gpu' has no attribute '{name}'")
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import redblackgraph as rb
from redblackgraph.gpu import rb_matrix_gpu
from scipy.sparse import csr_matrix

# Create sparse matrix on CPU
A_cpu = rb.matrix(csr_matrix([[rb.RED_ONE, 2, 0],
                               [0, rb.BLACK_ONE, 3],
                               [4, 0, 5]]))

# Transfer to GPU
A_gpu = rb_matrix_gpu(A_cpu)

# Compute on GPU
B_gpu = A_gpu @ A_gpu
C_gpu = B_gpu @ A_gpu

# Transfer result back
result = C_gpu.to_cpu()
```

### Batch Operations

```python
# Efficient: Keep data on GPU for multiple operations
A_gpu = rb_matrix_gpu(A_cpu)

# Compute A^8 without CPU transfers
result = A_gpu
for i in range(7):
    result = result @ A_gpu

final = result.to_cpu()
```

### Transitive Closure

```python
# Compute transitive closure on GPU
A_gpu = rb_matrix_gpu(genealogy_matrix)
A_star = A_gpu.transitive_closure()
relationships = A_star.to_cpu()
```

## Next Steps

Read **[04_performance_strategy.md](04_performance_strategy.md)** for optimization targets and benchmarking approach.
