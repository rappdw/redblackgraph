# GPU Implementation Architecture

## Design Philosophy

**Principle**: Build incrementally, validate constantly, optimize ruthlessly.

The GPU implementation should be:
1. **Modular** - Clean separation between CUDA, CuPy, and Python layers
2. **Compatible** - Drop-in replacement for existing `rb_matrix` API
3. **Correct** - Bit-exact results matching CPU implementation
4. **Fast** - Leverages GPU parallelism for sparse operations
5. **Maintainable** - Clear code, good documentation, comprehensive tests

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  Python API Layer (redblackgraph.gpu)                   │
│  - rb_matrix_gpu class                                  │
│  - API compatibility with rb_matrix                     │
│  - Memory management (CPU ↔ GPU)                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  CuPy Integration Layer                                 │
│  - Array wrapping and conversion                        │
│  - RawKernel loading and invocation                     │
│  - Memory allocation/deallocation                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  CUDA Kernel Layer                                      │
│  - avos_sum_kernel                                      │
│  - avos_product_kernel (with parity constraints)        │
│  - sparse_matmul_kernel (CSR format)                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GPU Hardware (CUDA-capable NVIDIA GPU)                 │
└─────────────────────────────────────────────────────────┘
```

## Module Structure

```
redblackgraph/
├── gpu/                          # New GPU backend
│   ├── __init__.py               # Public API exports
│   ├── rb_matrix_gpu.py          # Main GPU matrix class
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── avos_kernels.cu       # CUDA: avos_sum, avos_product
│   │   ├── sparse_kernels.cu     # CUDA: CSR sparse matmul
│   │   └── utils.cuh             # CUDA: shared utilities
│   ├── cupy_wrapper.py           # CuPy RawKernel management
│   ├── memory.py                 # Memory management utilities
│   └── tests/
│       ├── __init__.py
│       ├── test_avos_ops.py      # Test AVOS operations
│       ├── test_sparse_matmul.py # Test sparse multiplication
│       └── test_gpu_matrix.py    # Test rb_matrix_gpu API
├── [existing modules...]
```

## Key Classes

### `rb_matrix_gpu`

Main user-facing class, API-compatible with `rb_matrix`:

```python
class rb_matrix_gpu:
    """GPU-accelerated sparse Red-Black matrix.
    
    Attributes:
        data: CuPy array of non-zero values
        indices: CuPy array of column indices (CSR format)
        indptr: CuPy array of row pointers (CSR format)
        shape: Tuple of matrix dimensions
        dtype: NumPy dtype of elements
        device: GPU device ID
    """
    
    def __init__(self, matrix, device=0):
        """Create GPU matrix from scipy sparse or numpy array."""
        
    def __matmul__(self, other):
        """Matrix multiplication using AVOS semiring."""
        
    def to_cpu(self):
        """Transfer back to CPU as rb_matrix."""
        
    def transitive_closure(self):
        """Compute transitive closure on GPU."""
```

### `AVOSKernels`

Manages CUDA kernel loading and invocation:

```python
class AVOSKernels:
    """Manages CUDA kernels for AVOS operations.
    
    Lazy-loads kernels on first use and caches them.
    Handles different data types (int8, int16, int32, int64).
    """
    
    def avos_sum(self, x, y, out, dtype):
        """Element-wise AVOS sum."""
        
    def avos_product(self, x, y, out, dtype):
        """Element-wise AVOS product with parity constraints."""
        
    def sparse_matmul(self, A_data, A_indices, A_indptr,
                      B_data, B_indices, B_indptr,
                      C_data, C_indices, C_indptr, dtype):
        """Sparse matrix multiplication (CSR × CSR)."""
```

## Design Decisions

### 1. Why CuPy as Primary Interface?

**Decision**: Use CuPy as the main GPU array library.

**Rationale**:
- **NumPy-compatible API** - Familiar to users
- **RawKernel support** - Easy custom CUDA kernels
- **SciPy integration** - Already used by scipy.sparse
- **Active development** - Well-maintained, NumPy 2.x support
- **Good documentation** - Easy to learn and debug

**Alternatives considered**:
- **PyTorch**: Heavy dependency, different API paradigm
- **JAX**: Different execution model, less control
- **Raw CUDA + PyCUDA**: More boilerplate, harder to maintain

### 2. Sparse Format: CSR (Compressed Sparse Row)

**Decision**: Use CSR format as primary sparse representation.

**Rationale**:
- **Existing format** - Current CPU implementation uses CSR
- **Row-based access** - Good for matrix multiplication
- **cuSPARSE compatible** - Can leverage NVIDIA's library if needed
- **Memory efficient** - Genealogy graphs are very sparse

**CSR Structure**:
```
data:    [val₁, val₂, ..., valₙₙz]  # Non-zero values
indices: [col₁, col₂, ..., colₙₙz]  # Column index for each value
indptr:  [ptr₀, ptr₁, ..., ptrₙᵣₒᵥₛ₊₁] # Start index of each row
```

### 3. Two-Pass Sparse Multiplication Algorithm

**Decision**: Implement the same two-pass algorithm as CPU version.

**Rationale**:
- **Known correctness** - Already tested and validated
- **Predictable memory** - Pre-allocate exact size needed
- **GPU-friendly** - Both passes parallelize well

**Algorithm**:
```
Pass 1: Compute output row pointers
  - For each output row i:
    - Count non-zeros in row i of result
    - Store count in C_indptr
  - Prefix sum to get CSR indptr array

Pass 2: Compute output values and indices
  - For each output row i:
    - For each non-zero in result row:
      - Compute value using AVOS operations
      - Store in C_data and C_indices
```

### 4. Memory Management Strategy

**Decision**: Explicit memory management with automatic transfer.

**Rationale**:
- **Control** - User can keep matrices on GPU for multiple operations
- **Convenience** - Automatic transfer for single operations
- **Performance** - Minimize CPU↔GPU transfers

**API Pattern**:
```python
# Automatic (single operation)
A_gpu = rb_matrix_gpu(A_cpu)
C_cpu = (A_gpu @ A_gpu).to_cpu()

# Manual (multiple operations)
A_gpu = rb_matrix_gpu(A_cpu)
B_gpu = A_gpu @ A_gpu
C_gpu = B_gpu @ A_gpu
result = C_gpu.to_cpu()
```

### 5. Type Support

**Decision**: Support int8, int16, int32, int64 (same as CPU).

**Rationale**:
- **Compatibility** - Match existing CPU implementation
- **Use cases** - Different size trees need different ranges
- **Templates** - CUDA templates handle multiple types cleanly

**Implementation**:
- Use C++ templates in CUDA kernels
- Python side dispatches based on dtype
- No automatic type promotion (fail fast on mismatch)

### 6. Error Handling

**Decision**: Python-side validation, CUDA-side asserts.

**Rationale**:
- **Early detection** - Catch errors before GPU launch
- **Better messages** - Python can provide context
- **Performance** - No runtime checks in hot paths

**Strategy**:
```python
# Python side: validate inputs
if A.shape[1] != B.shape[0]:
    raise ValueError(f"Incompatible shapes: {A.shape} @ {B.shape}")

# CUDA side: asserts for invariants
assert(threadIdx.x < blockDim.x);
assert(row_idx < num_rows);
```

## Integration with Existing Code

### Backward Compatibility

The GPU implementation should be **opt-in**:

```python
# Existing code continues to work
A = rb_matrix(scipy_sparse)
C = A @ A  # Uses CPU

# GPU acceleration via explicit call
A_gpu = rb_matrix_gpu(A)
C = (A_gpu @ A_gpu).to_cpu()

# Or via method
C = A.gpu() @ A.gpu()  # If we add .gpu() method
```

### Sharing Test Infrastructure

Leverage existing tests by running them on GPU:

```python
# tests/gpu/test_gpu_matrix.py
import tests.sparse.test_sparse_matmul as cpu_tests

class TestGPUMatmul(cpu_tests.TestSparseMatmul):
    """Run CPU tests on GPU implementation."""
    
    def setup_method(self):
        # Override to use GPU implementation
        self.matrix_class = rb_matrix_gpu
```

## Build System Integration

### Meson Configuration

Add GPU option to `meson.build`:

```meson
# Check for CUDA
cuda_available = add_languages('cuda', required: false)

if cuda_available
  cuda_compiler = meson.get_compiler('cuda')
  
  # GPU module
  py.extension_module(
    '_gpu_kernels',
    'redblackgraph/gpu/kernels/avos_kernels.cu',
    'redblackgraph/gpu/kernels/sparse_kernels.cu',
    dependencies: [py_dep, cuda_dep],
    install: true,
  )
endif
```

### Optional Dependency

Make GPU support optional:

```python
# redblackgraph/gpu/__init__.py
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

if CUPY_AVAILABLE:
    from .rb_matrix_gpu import rb_matrix_gpu
    __all__ = ['rb_matrix_gpu']
else:
    __all__ = []
    
def __getattr__(name):
    if not CUPY_AVAILABLE:
        raise ImportError(
            f"GPU support requires CuPy. "
            f"Install with: pip install cupy-cuda11x"
        )
```

## Compatibility Matrix

| Python | NumPy | SciPy | CuPy | CUDA | Status |
|--------|-------|-------|------|------|--------|
| 3.10   | 2.0+  | 1.11+ | 12.0+| 11.0+| Target |
| 3.11   | 2.0+  | 1.11+ | 12.0+| 11.0+| Target |
| 3.12   | 2.0+  | 1.11+ | 12.0+| 11.0+| Target |
| 3.13   | 2.2+  | 1.14+ | 13.0+| 11.0+| Future |

## Performance Considerations

### When to Use GPU

GPU acceleration helps when:
- **Matrix size** > 1000×1000
- **Multiple operations** on same matrix
- **Batch operations** on multiple matrices
- **Transitive closure** on large graphs

GPU may be **slower** when:
- Matrix size < 100×100 (transfer overhead)
- Single operation (transfer cost dominates)
- Very dense matrices (better done on CPU)

### Memory Constraints

Typical GPU memory (8-16GB) can handle:
- **Dense**: Up to ~30K × 30K int32 matrices
- **Sparse (1% density)**: Up to ~300K × 300K matrices
- **Genealogy graphs (0.1%)**: Up to ~1M × 1M matrices

## Security and Safety

1. **Input validation** - Check array bounds before GPU launch
2. **Memory safety** - Use RAII patterns for GPU memory
3. **Error propagation** - Catch and report CUDA errors clearly
4. **Resource cleanup** - Ensure GPU memory is freed on exceptions

## Next Steps

1. Read **[02_cuda_kernels.md](02_cuda_kernels.md)** for kernel implementation details
2. Review **[03_cupy_integration.md](03_cupy_integration.md)** for Python API design
3. Check **[04_performance_strategy.md](04_performance_strategy.md)** for optimization approach
