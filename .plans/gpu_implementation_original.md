# GPU Implementation Plan for Avos Matrix Multiplication

## Current State

**Avos Operations** (custom semiring for Red-Black graphs):
- **avos_sum**: Non-zero minimum `min(a, b)` where 0 is identity
- **avos_product**: Bit manipulation - replaces MSB of `rhs` with `lhs`

**Existing Implementations**:
1. **`redblackgraph.reference`**: Pure Python (illustrative)
2. **`redblackgraph.core`**: C/NumPy extension with einsum
3. **`redblackgraph.sparse`**: CSR sparse matrices (C++ extension)

**Use Case**: Very sparse matrices representing genealogy DAGs

---

## GPU Implementation Plan

### **Phase 1: Module Structure & Setup**

**Create new module**: `redblackgraph/gpu/`

**Key components**:
- Python interface layer (`__init__.py`, `rbm_gpu.py`)
- CUDA kernels (`kernels/avos_kernels.cu`)
- CuPy/CUDA integration
- Sparse format support (CSR initially)

**Dependencies to add**:
- CuPy (for GPU array operations)
- cuSPARSE (for sparse matrix operations)
- Optional: PyTorch or JAX for alternative backends

---

### **Phase 2: CUDA Kernel Implementation**

**Core CUDA kernels needed**:

1. **`avos_sum_kernel`**: Element-wise avos sum
   - Simple comparison logic
   - Handles zero identity properly
   
2. **`avos_product_kernel`**: Bit manipulation on GPU
   - MSB computation using `__clz()` (count leading zeros)
   - Bit shifting and masking
   - Overflow detection

3. **`sparse_matmul_kernel`**: Sparse matrix multiplication
   - CSR format: row pointers, column indices, values
   - Two-pass algorithm (matching current CPU implementation):
     - **Pass 1**: Compute output row pointers
     - **Pass 2**: Compute values using avos operations
   - Thread per row or warp per row strategies

---

### **Phase 3: Sparse Matrix Multiplication Strategy**

**For matrices fitting in GPU memory**:

**Option A: cuSPARSE + Custom Operations** (Recommended first approach)
- Use cuSPARSE structure and utilities
- Implement custom semiring via callback or kernel fusion
- Leverage existing CSR infrastructure

**Option B: Custom SpGEMM Implementation**
- Implement specialized sparse-sparse matmul
- Optimize for genealogy graph characteristics:
  - Very sparse (few edges per vertex)
  - DAG structure (no cycles)
  - Small local neighborhoods
- Use hash-based accumulation for intermediate results

**Specific optimizations**:
- **Warp-level operations**: Use warp shuffle for reductions
- **Shared memory**: Cache rows/columns
- **Memory coalescing**: Ensure aligned access patterns
- **Occupancy tuning**: Adjust threads per block

---

### **Phase 4: Python Interface**

**API Design** (matching existing `rb_matrix`):

```python
from redblackgraph.gpu import rb_matrix_gpu

# Create from scipy sparse or numpy
A_gpu = rb_matrix_gpu(A_csr)  

# Matrix multiplication with @ operator
C = A_gpu @ A_gpu

# Transfer back to CPU
C_cpu = C.to_cpu()  # Returns rb_matrix

# Support for transitive closure on GPU
C_star = A_gpu.transitive_closure()
```

**Memory management**:
- Automatic CPU ↔ GPU transfers
- Option to keep data on GPU for multiple operations
- Context manager for explicit control

---

### **Phase 5: Testing & Validation**

**Test strategy**:
1. **Unit tests**: Compare GPU vs CPU avos operations
2. **Small matrices**: Verify against reference implementation
3. **Existing test suite**: Run all sparse tests with GPU backend
4. **Numerical accuracy**: Verify bit-exact results
5. **Edge cases**: Overflow handling, zero matrices, identity

**Benchmarking matrices**:
- Range from 100x100 to 100,000x100,000
- Various sparsity levels (0.1% - 5%)
- Real genealogy datasets from tests

---

### **Phase 6: Performance Optimization**

**Profiling tools**:
- NVIDIA Nsight Systems/Compute
- CuPy profiler
- Custom timing utilities

**Optimization targets**:
1. **Kernel occupancy**: Maximize GPU utilization
2. **Memory bandwidth**: Minimize transfers
3. **Register usage**: Balance parallelism
4. **Kernel fusion**: Combine operations where possible

**Expected speedups** (for sparse matrices):
- Small matrices (<1000×1000): 2-5x
- Medium matrices (1000-10000): 5-20x
- Large matrices (>10000): 10-50x

---

### **Future: Out-of-Core Support**

**For matrices exceeding GPU memory** (Phase 7+):
- **Matrix tiling**: Process in blocks
- **Streaming**: Overlap compute and transfer
- **CPU-GPU hybrid**: Keep hot data on GPU
- **Multi-GPU**: Distribute across GPUs on DGX

---

## Implementation Priority

### **Immediate (Phase 1-3)**:
1. Set up `redblackgraph/gpu/` module structure
2. Implement basic CUDA kernels for avos operations
3. Create CSR sparse matrix multiplication kernel
4. Python wrapper with CuPy integration

### **Short-term (Phase 4-5)**:
1. Complete API matching existing `rb_matrix`
2. Comprehensive testing against CPU implementation
3. Integration with existing codebase

### **Medium-term (Phase 6+)**:
1. Performance optimization and profiling
2. Multi-GPU support
3. Out-of-core algorithms for huge matrices

---

## Technical Considerations

**Data Types**: Support int8, int16, int32, int64 (matching CPU)

**Memory Layout**: CSR format is GPU-friendly for row-based operations

**Synchronization**: Minimize CPU-GPU synchronization points

**Error Handling**: Proper overflow detection in avos_product

**Portability**: Consider CuPy for maintainability vs raw CUDA for performance

---

## Recommended First Steps

1. **Create module structure**:
   ```
   redblackgraph/gpu/
   ├── __init__.py
   ├── rbm_gpu.py          # Python interface
   ├── avos_ops.py         # CuPy kernels for avos operations
   ├── sparse_matmul.py    # Sparse multiplication
   ├── kernels/
   │   ├── avos.cu         # CUDA kernels
   │   └── sparse.cu       # Sparse kernels
   ├── setup.py            # Build configuration
   └── tests/
       └── test_gpu.py     # GPU-specific tests
   ```

2. **Start with CuPy RawKernels**: Faster prototyping than pure CUDA

3. **Verify correctness first**: Match CPU results exactly before optimizing

4. **Incremental benchmarking**: Profile after each optimization

---

## Repository Analysis Summary

### Avos Operations Details

From `redblackgraph/reference/rbg_math.py`:
```python
def avos_sum(x: int, y: int) -> int:
    '''The avos sum is the non-zero minimum of x and y'''
    if x == 0: return y
    if y == 0: return x
    if x < y: return x
    return y

def avos_product(x: int, y: int) -> int:
    '''The avos product replaces the left most significant bit of operand 2 with operand 1'''
    if x == 0 or y == 0: return 0
    # Special case handling for -1
    if x == -1:
        if y == 1: return -1
        x = 1
    if y == -1:
        if x == 1: return -1
        y = 1
    
    bit_position = MSB(y)
    return ((y & (2 ** bit_position - 1)) | (x << bit_position))
```

### Matrix Multiplication Pattern

From `redblackgraph/reference/mat_avos.py`:
```python
def mat_avos(A, B):
    return [[reduce(avos_sum, [avos_product(a, b) for a, b in zip(A_row, B_col)]) 
             for B_col in zip(*B)] 
            for A_row in A]
```

This is equivalent to standard matrix multiplication but with:
- Addition replaced by `avos_sum`
- Multiplication replaced by `avos_product`

### Current Sparse Implementation

The CPU sparse implementation (`redblackgraph/sparse/`) uses:
- CSR (Compressed Sparse Row) format
- Two-pass algorithm for sparse matrix multiplication
- C++ implementation in `sparsetools/rbm_math.h`

Key characteristics:
- Very sparse matrices (genealogy graphs typically have ~2-4 edges per vertex)
- Integer data types (int8 through int64)
- Supports transitive closure computation
- Used for family history relationship calculations
