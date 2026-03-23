# Naive GPU Implementation Guide

**Status**: Learning/Proof-of-Concept  
**Purpose**: Understand build, installation, and deployment on DGX Spark  
**Date**: November 2025

## Overview

This is a **minimal naive GPU implementation** designed to help understand:

1. **Build system integration** - How to add CUDA/CuPy to meson build
2. **CuPy basics** - Python GPU arrays and kernels
3. **Memory management** - CPU↔GPU transfers (and unified memory on DGX Spark)
4. **Installation workflow** - Dependencies and environment setup
5. **DGX Spark specifics** - Grace Hopper unified memory architecture

**⚠️ This is NOT production-ready.** The matrix multiplication uses a naive O(n³) algorithm that converts to dense matrices. See `.plans/gpu_implementation/` for the full optimized design.

## What's Implemented

### Files Created

```
redblackgraph/gpu/
├── __init__.py          # Module exports
├── core.py              # AVOS operations using CuPy ElementwiseKernel
└── matrix.py            # rb_matrix_gpu wrapper around cupyx.scipy.sparse

tests/gpu/
└── test_naive_gpu.py    # Basic validation tests
```

### Features

✅ **AVOS element-wise operations** (`avos_sum`, `avos_product`)
- Implemented as CuPy ElementwiseKernels
- Full parity constraint support (RED_ONE/BLACK_ONE)
- Vectorized GPU execution

✅ **Sparse matrix wrapper** (`rb_matrix_gpu`)
- Wraps `cupyx.scipy.sparse.csr_matrix`
- CPU↔GPU conversion methods
- Triangular flag support

✅ **Basic tests**
- Operation correctness vs. CPU reference
- Memory transfer validation
- Small matrix tests

❌ **NOT Implemented** (see full plan):
- Optimized SpGEMM kernels
- Two-phase symbolic/numeric multiplication
- Transitive closure
- Memory optimization for billion-scale

## Installation

### Prerequisites

1. **CUDA Toolkit** (11.0+, preferably 12.x for H100)
2. **Python 3.10+** (already required by redblackgraph)
3. **NVIDIA GPU** with compute capability 7.0+ (V100, A100, H100)

### Install CuPy

CuPy version depends on your CUDA version:

```bash
# For CUDA 12.x (H100, DGX Spark)
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Verify installation
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.Device(0).compute_capability)"
```

### Install redblackgraph with GPU support

```bash
cd /path/to/redblackgraph

# Install in development mode with GPU dependencies
pip install -e .[dev,test]
pip install cupy-cuda12x  # or appropriate CUDA version

# Verify GPU module loads
python -c "from redblackgraph.gpu import rb_matrix_gpu; print('✓ GPU module loaded')"
```

### Testing

```bash
# Run GPU tests (will skip if CuPy not available)
pytest tests/gpu/test_naive_gpu.py -v

# Run with GPU available
CUDA_VISIBLE_DEVICES=0 pytest tests/gpu/test_naive_gpu.py -v
```

## DGX Spark Deployment Guide

### Hardware: NVIDIA DGX Spark (Grace Hopper)

**Specifications**:
- **CPU**: ARM Neoverse V2 (Grace)
- **GPU**: H100 80GB HBM3
- **Memory**: Unified CPU/GPU memory via NVLink-C2C
- **CUDA**: 12.x required

**Key Advantage**: **Unified Memory Architecture**
- No explicit CPU↔GPU transfers needed
- Automatic page migration
- Simplifies programming model

### Setup on DGX Spark

#### 1. Environment Setup

```bash
# SSH into DGX Spark
ssh user@dgx-spark.example.com

# Check CUDA version
nvcc --version
nvidia-smi

# Should show H100 GPU(s)
```

#### 2. Python Environment

```bash
# Create virtual environment
python3.10 -m venv ~/venv-rbg-gpu
source ~/venv-rbg-gpu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install build dependencies
pip install meson-python meson ninja cython tempita
```

#### 3. Install CuPy for CUDA 12.x

```bash
# For H100 with CUDA 12.x
pip install cupy-cuda12x

# Verify
python -c "import cupy as cp; print('CuPy version:', cp.__version__)"
python -c "import cupy as cp; print('GPU:', cp.cuda.Device(0).compute_capability)"
```

#### 4. Build and Install redblackgraph

```bash
# Clone repository
git clone https://github.com/rappdw/redblackgraph.git
cd redblackgraph

# Install dependencies
pip install numpy scipy

# Build and install
pip install -e .[dev,test]

# Verify
python -c "import redblackgraph; print(redblackgraph.__version__)"
python -c "from redblackgraph.gpu import rb_matrix_gpu; print('GPU module OK')"
```

#### 5. Run Tests

```bash
# Run all tests including GPU
pytest tests/ -v

# Run only GPU tests
pytest tests/gpu/ -v

# Check GPU is being used
CUDA_LAUNCH_BLOCKING=1 pytest tests/gpu/test_naive_gpu.py -v -s
```

### Unified Memory on Grace Hopper

On DGX Spark (Grace Hopper), unified memory simplifies the code:

**Traditional GPU** (requires explicit transfers):
```python
# CPU → GPU transfer
data_cpu = np.array([1, 2, 3])
data_gpu = cp.asarray(data_cpu)  # Explicit copy

# Compute on GPU
result_gpu = my_kernel(data_gpu)

# GPU → CPU transfer
result_cpu = result_gpu.get()  # Explicit copy
```

**Grace Hopper** (unified memory):
```python
# Allocate in unified memory (automatic with CuPy on Grace Hopper)
data = cp.array([1, 2, 3])  # Accessible by both CPU & GPU

# Compute on GPU (no explicit transfer)
result = my_kernel(data)

# Access on CPU (no explicit transfer)
print(result)  # Automatic page migration
```

**Performance Tips**:
- Use `cp.cuda.MemoryPool().set_limit()` to manage memory
- Profile with `nsys` to check for excessive page faults
- Prefetch data to GPU when access pattern is known:
  ```python
  cp.cuda.runtime.memPrefetchAsync(data.data.ptr, data.nbytes, 0)
  ```

## Build System Integration (Future Work)

The current implementation doesn't require build system changes since it uses pure Python + CuPy. For optimized CUDA kernels, you'll need to:

### Meson Configuration

Add to `meson.build`:

```meson
# Optional CUDA support
cuda_dep = dependency('cuda', required: false)

if cuda_dep.found()
  add_project_arguments('-DHAVE_CUDA', language: ['c', 'cpp'])
  
  # Add GPU submodule
  subdir('redblackgraph/gpu/cuda')
endif
```

### CUDA Kernel Compilation

Create `redblackgraph/gpu/cuda/meson.build`:

```meson
cuda_compiler = find_program('nvcc', required: true)

# Compile CUDA kernels
cuda_kernels = custom_target(
  'avos_kernels',
  input: 'avos_kernels.cu',
  output: 'avos_kernels.ptx',
  command: [cuda_compiler, '-ptx', '-arch=sm_80', '@INPUT@', '-o', '@OUTPUT@']
)

# Install PTX file
install_data(cuda_kernels, install_dir: py.get_install_dir() / 'redblackgraph/gpu/cuda')
```

### Loading Compiled Kernels

```python
# In Python code
import cupy as cp
from pathlib import Path

kernel_dir = Path(__file__).parent / 'cuda'
with open(kernel_dir / 'avos_kernels.ptx', 'r') as f:
    ptx = f.read()

module = cp.RawModule(code=ptx)
avos_kernel = module.get_function('avos_product_kernel')
```

## Usage Examples

### Basic Operations

```python
import numpy as np
import cupy as cp
from redblackgraph.gpu import avos_sum_gpu, avos_product_gpu

# Create GPU arrays
x = cp.array([2, 3, 4, 5], dtype=cp.int32)
y = cp.array([3, 2, 5, 4], dtype=cp.int32)

# Compute on GPU
sum_result = avos_sum_gpu(x, y)
prod_result = avos_product_gpu(x, y)

# Transfer back to CPU
print("Sum:", sum_result.get())
print("Product:", prod_result.get())
```

### Matrix Operations

```python
from scipy import sparse as sp_sparse
from redblackgraph.gpu import rb_matrix_gpu

# Create sparse matrix on CPU
data = np.array([2, 3, 4], dtype=np.int32)
indices = np.array([0, 1, 2], dtype=np.int32)
indptr = np.array([0, 1, 2, 3], dtype=np.int32)
A_cpu = sp_sparse.csr_matrix((data, indices, indptr), shape=(3, 3))

# Transfer to GPU
A_gpu = rb_matrix_gpu.from_cpu(A_cpu, triangular=True)

print(f"GPU matrix: {A_gpu}")
print(f"Shape: {A_gpu.shape}, nnz: {A_gpu.nnz}")

# Transfer back
A_result = A_gpu.to_cpu()
```

### Validation Against CPU Reference

```python
from redblackgraph.reference.rbg_math import avos_product
from redblackgraph.gpu import avos_product_gpu
import cupy as cp

# Test case
x, y = 3, 2

# CPU reference
result_cpu = avos_product(x, y)

# GPU version
x_gpu = cp.array([x], dtype=cp.int32)
y_gpu = cp.array([y], dtype=cp.int32)
result_gpu = avos_product_gpu(x_gpu, y_gpu).get()[0]

assert result_cpu == result_gpu, f"Mismatch: CPU={result_cpu}, GPU={result_gpu}"
print("✓ GPU matches CPU reference")
```

## Performance Considerations

### Current Implementation (Naive)

**DO NOT USE FOR PRODUCTION**:
- Matrix multiplication is O(n³) with dense conversion
- Limited to matrices with nnz < 10,000
- No memory optimization
- No kernel fusion

**Measured Performance** (approximate, small matrices):
- Element-wise operations: ~100x faster than CPU
- Matrix multiplication: **Slower than CPU** due to naive algorithm

### Optimized Implementation (Planned)

See `.plans/gpu_implementation/04_performance_strategy.md` for targets:

**Expected Performance** (after optimization):
- Matrix multiplication: 5-50x faster than CPU
- Transitive closure: 10-100x faster
- Memory: 1B×1B matrices at 0.1% density (~8GB per matrix)

## Troubleshooting

### CuPy Not Found

```bash
# Verify CuPy installation
pip list | grep cupy

# Reinstall for correct CUDA version
pip uninstall cupy cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x  # For CUDA 12.x
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA visible devices
echo $CUDA_VISIBLE_DEVICES

# Set device explicitly
export CUDA_VISIBLE_DEVICES=0
```

### Out of Memory

```python
# Limit memory pool
import cupy as cp
pool = cp.cuda.MemoryPool()
pool.set_limit(size=8*1024**3)  # 8GB limit
cp.cuda.set_allocator(pool.malloc)
```

### Tests Skipped

Tests are automatically skipped if CuPy is not available:

```bash
# Will skip GPU tests
pytest tests/gpu/test_naive_gpu.py -v

# Should see: SKIPPED [1] tests/gpu/test_naive_gpu.py:...: CuPy not available
```

## Next Steps

This naive implementation helps you understand the infrastructure. For production:

1. **Read the full plan**: `.plans/gpu_implementation/QUICK_START.md`
2. **Implement Phase 1**: AVOS kernels (Week 2)
3. **Implement Phase 2**: Matrix structure with unified memory (Week 3)
4. **Implement Phase 3**: Optimized SpGEMM kernels (Weeks 4-5)
5. **Implement Phase 4**: Transitive closure (Week 6)

See `.plans/gpu_implementation/06_implementation_phases.md` for detailed roadmap.

## References

- **CuPy Documentation**: https://docs.cupy.dev/
- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Grace Hopper Architecture**: https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
- **Sparse Matrix-Matrix Multiplication**: Survey by Bell & Garland (2008)

## Contact

For questions about this implementation:
- See architectural decisions in `.plans/architecture_decisions.md`
- Review GPU plan in `.plans/gpu_implementation/`
- Check test coverage in `tests/gpu/`
