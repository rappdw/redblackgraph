"""
GPU-accelerated rb_matrix using CuPy sparse matrices.

This is a naive implementation that:
1. Wraps cupyx.scipy.sparse.csr_matrix
2. Implements basic @ operator using element-wise kernels (not optimized)
3. Provides conversion to/from CPU rb_matrix
"""

try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_sparse = None

import numpy as np
from .core import avos_sum_gpu, avos_product_gpu, _check_cupy


class rb_matrix_gpu:
    """
    GPU-accelerated Red-Black matrix using CuPy sparse CSR format.
    
    This is a minimal naive implementation for learning purposes.
    
    Features:
    - Wraps cupyx.scipy.sparse.csr_matrix
    - Supports triangular=True flag for upper triangular matrices
    - Implements basic @ operator (naive, not optimized)
    - Provides to_cpu() and from_cpu() conversions
    
    NOT YET IMPLEMENTED:
    - Optimized SpGEMM kernels
    - Two-phase symbolic/numeric multiplication
    - Transitive closure
    - Advanced memory management
    """
    
    def __init__(self, data=None, *, triangular=False, shape=None):
        """
        Initialize GPU rb_matrix.
        
        Args:
            data: Can be:
                - cupyx.scipy.sparse matrix
                - CPU scipy.sparse matrix (will be transferred)
                - (data, indices, indptr) tuple for CSR format
                - None (creates empty matrix, requires shape)
            triangular: If True, matrix is upper triangular
            shape: Shape tuple (required if data is None)
        """
        _check_cupy()
        
        self.triangular = triangular
        
        if data is None:
            if shape is None:
                raise ValueError("shape required when data is None")
            # Create empty CSR matrix
            self.data = cp_sparse.csr_matrix(shape, dtype=cp.float32)
        elif isinstance(data, cp_sparse.csr_matrix):
            self.data = data
        elif hasattr(data, 'tocsr'):
            # Convert from other sparse format or CPU sparse matrix
            # Transfer to GPU if needed
            if hasattr(data, 'get'):
                # Already on GPU
                self.data = data.tocsr()
            else:
                # CPU sparse matrix - convert to CuPy
                cpu_csr = data.tocsr()
                self.data = cp_sparse.csr_matrix(
                    (cp.array(cpu_csr.data, dtype=cp.float32),
                     cp.array(cpu_csr.indices, dtype=cp.int32),
                     cp.array(cpu_csr.indptr, dtype=cp.int32)),
                    shape=cpu_csr.shape
                )
        elif isinstance(data, tuple) and len(data) == 3:
            # CSR format: (data, indices, indptr)
            data_arr, indices_arr, indptr_arr = data
            self.data = cp_sparse.csr_matrix(
                (cp.asarray(data_arr, dtype=cp.float32),
                 cp.asarray(indices_arr, dtype=cp.int32),
                 cp.asarray(indptr_arr, dtype=cp.int32)),
                shape=shape
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @property
    def shape(self):
        """Matrix shape."""
        return self.data.shape
    
    @property
    def nnz(self):
        """Number of non-zero elements."""
        return self.data.nnz
    
    def to_cpu(self):
        """
        Transfer matrix to CPU.
        
        Returns:
            scipy.sparse.csr_matrix on CPU
        """
        # Transfer GPU arrays to CPU
        from scipy import sparse as sp_sparse
        
        return sp_sparse.csr_matrix(
            (self.data.data.get(),
             self.data.indices.get(),
             self.data.indptr.get()),
            shape=self.shape
        )
    
    @classmethod
    def from_cpu(cls, cpu_matrix, triangular=False):
        """
        Create GPU matrix from CPU sparse matrix.
        
        Args:
            cpu_matrix: scipy.sparse matrix
            triangular: If True, matrix is upper triangular
            
        Returns:
            rb_matrix_gpu instance
        """
        return cls(cpu_matrix, triangular=triangular)
    
    def __matmul__(self, other):
        """
        Matrix multiplication using AVOS operations.
        
        NAIVE IMPLEMENTATION: Uses element-wise kernels, not optimized SpGEMM.
        
        This is for learning/testing only. Production implementation would use
        custom two-phase SpGEMM kernels as described in the plan.
        """
        if not isinstance(other, rb_matrix_gpu):
            raise TypeError("Can only multiply rb_matrix_gpu with rb_matrix_gpu")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        
        # NAIVE APPROACH: Convert to dense, multiply, convert back
        # This is EXTREMELY inefficient and only for demonstration
        # Real implementation would use custom SpGEMM kernels
        
        import warnings
        warnings.warn(
            "Using naive dense matrix multiplication. "
            "This is for learning only and will be slow/memory-intensive!",
            UserWarning,
            stacklevel=2
        )
        
        # For very small matrices only
        if self.nnz > 10000 or other.nnz > 10000:
            raise NotImplementedError(
                "Naive implementation limited to small matrices (nnz < 10k). "
                "Use optimized SpGEMM kernels for production."
            )
        
        # Convert to dense
        A_dense = self.data.toarray()
        B_dense = other.data.toarray()
        
        # Result matrix
        C_dense = cp.zeros((self.shape[0], other.shape[1]), dtype=cp.float32)
        
        # Naive triple loop (extremely slow, for demonstration only)
        m, n, p = self.shape[0], self.shape[1], other.shape[1]
        
        for i in range(m):
            for j in range(p):
                # Skip if triangular and j < i
                if self.triangular and j < i:
                    continue
                
                # Compute C[i,j] = sum over k of avos_product(A[i,k], B[k,j])
                acc = cp.float32(0)
                for k in range(n):
                    if A_dense[i, k] != 0 and B_dense[k, j] != 0:
                        prod = avos_product_gpu(A_dense[i, k], B_dense[k, j])
                        acc = avos_sum_gpu(acc, prod)
                C_dense[i, j] = acc
        
        # Convert back to sparse
        C_sparse = cp_sparse.csr_matrix(C_dense)
        
        return rb_matrix_gpu(C_sparse, triangular=self.triangular)
    
    def __repr__(self):
        return (f"rb_matrix_gpu(shape={self.shape}, nnz={self.nnz}, "
                f"triangular={self.triangular}, device='gpu')")
