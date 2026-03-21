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
├── transitive_closure.py  # GPU-resident transitive closure via repeated squaring
└── README.md              # This file
```

## Quick Usage

```python
import numpy as np
import scipy.sparse as sp
from redblackgraph.gpu import CSRMatrixGPU, spgemm, transitive_closure_gpu

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

# Transitive closure
R_gpu, diameter = A_gpu.transitive_closure()
R_cpu = R_gpu.to_cpu()
```

## Dependencies

```bash
pip install cupy-cuda12x  # or cupy-cuda11x
```

GPU tests are skipped automatically if CuPy is unavailable.

## Testing

```bash
pytest tests/gpu/ -v
```
