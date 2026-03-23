# GPU Module

GPU-accelerated AVOS operations using CuPy and CUDA.

## Module Structure

```
redblackgraph/gpu/
├── __init__.py            # Module exports + NVRTC probe
├── csr_gpu.py             # CSRMatrixGPU: sparse matrix on GPU with raw int32 buffers
├── avos_kernels.py        # CUDA RawKernels for AVOS sum/product (int32)
├── spgemm_symbolic.py     # Symbolic phase: compute output sparsity pattern via hash tables
├── spgemm_numeric.py      # Numeric phase: compute AVOS values with atomicMin
├── spgemm.py              # High-level SpGEMM API: spgemm(A, B)
├── transitive_closure.py  # GPU transitive closure: repeated squaring + DAG-specific
└── README.md              # This file
```

## Quick Usage

```python
import numpy as np
import scipy.sparse as sp
from redblackgraph.gpu import (
    CSRMatrixGPU, spgemm, transitive_closure_gpu, transitive_closure_dag_gpu
)

# Create GPU matrix from CPU sparse matrix
A_cpu = sp.csr_matrix(np.array([
    [1, 2, 0],
    [0, -1, 3],
    [0, 0, 1],
], dtype=np.int32))

A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)

# SpGEMM: C = A @ B
C_gpu = A_gpu @ A_gpu       # operator form
C_gpu = spgemm(A_gpu)       # function form (self-multiply)
C_gpu = spgemm(A_gpu, B_gpu)  # general A @ B

# Transitive closure — repeated squaring (any graph)
R_gpu, diameter = transitive_closure_gpu(A_gpu)

# Transitive closure — level-parallel DAG propagation (DAGs only, faster)
R_gpu, diameter = transitive_closure_dag_gpu(A_gpu)

R_cpu = R_gpu.to_cpu()
```

## Transitive Closure Algorithms

### Repeated squaring (`transitive_closure_gpu`)
Computes TC(A) = A + A² + A⁴ + A⁸ + ... using AVOS SpGEMM. Works for any graph.
Converges in O(log d) iterations where d is the graph diameter. All data stays
GPU-resident between iterations.

### Level-parallel DAG propagation (`transitive_closure_dag_gpu`)
Specialized for DAGs (triangular matrices). Computes topological levels, then
processes each level in parallel on GPU. At each level, a CUDA kernel expands
successor closures (applying avos_product), followed by sort + AVOS-sum reduction.

The DAG kernel achieves up to 10x speedup over the optimized Cython CPU algorithm
for graphs above ~1,000 vertices, nearly doubling the speedup of repeated squaring.

## Dependencies

```bash
pip install cupy-cuda12x  # or cupy-cuda11x
```

GPU tests are skipped automatically if CuPy is unavailable.

## Testing

```bash
pytest tests/gpu/ -v
```
