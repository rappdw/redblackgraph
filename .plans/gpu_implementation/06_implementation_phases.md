# Implementation Phases

## Overview

This document provides a step-by-step roadmap for implementing GPU acceleration for redblackgraph. Each phase builds on the previous one, with clear deliverables and validation criteria.

**Total Estimated Time**: 8-12 weeks

## Phase 0: Preparation (Week 1)

### Goals
- Set up development environment
- Validate dependencies
- Create project structure

### Tasks

#### 1.1 Environment Setup
```bash
# Create GPU development environment
conda create -n rbg-gpu python=3.11
conda activate rbg-gpu

# Install core dependencies
pip install numpy scipy meson ninja cython
pip install cupy-cuda12x  # Or cuda11x for CUDA 11

# Install redblackgraph in dev mode
pip install -e . --no-build-isolation

# Verify GPU access
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

#### 1.2 Create Module Structure
```bash
cd redblackgraph
mkdir -p gpu/{kernels,tests,benchmarks}

# Create initial files
touch gpu/__init__.py
touch gpu/rb_matrix_gpu.py
touch gpu/kernels/{__init__.py,avos_kernels.cu,sparse_kernels.cu}
touch gpu/tests/{__init__.py,conftest.py}
```

#### 1.3 Set Up Test Infrastructure
```python
# redblackgraph/gpu/tests/conftest.py
import pytest

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")

@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU not available."""
    try:
        import cupy as cp
        if not cp.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("CuPy not installed")
```

### Deliverables
- [x] GPU-enabled development environment
- [x] Module structure created
- [x] Basic test infrastructure
- [x] Documentation of setup process

### Validation
```bash
# Verify setup
python -c "import cupy as cp; print(f'CuPy {cp.__version__}, CUDA {cp.cuda.runtime.runtimeGetVersion()}')"
pytest redblackgraph/gpu/tests/ --collect-only
```

---

## Phase 1: Element-wise Operations (Week 2)

### Goals
- Implement AVOS sum and product kernels
- Create CuPy wrapper
- Test against CPU reference

### Tasks

#### 1.1 Implement AVOS Kernels

Create `redblackgraph/gpu/kernels/avos_kernels.cu`:
```cuda
// Implement:
// - avos_sum device function
// - avos_product device function (with parity constraints)
// - msb_position helper
// - Kernel launchers for element-wise operations
```

#### 1.2 Create Python Wrapper

Create `redblackgraph/gpu/kernels/__init__.py`:
```python
# Implement AVOSKernels class
# - Lazy kernel loading
# - Dtype dispatch
# - avos_sum() and avos_product() functions
```

#### 1.3 Write Unit Tests

Create `redblackgraph/gpu/tests/test_avos_kernels.py`:
```python
# Test:
# - Basic operations
# - Identity properties
# - Parity constraints
# - Random inputs vs reference
```

### Deliverables
- [x] Working CUDA kernels for AVOS operations
- [x] Python wrapper with CuPy RawKernel
- [x] Comprehensive unit tests (>20 tests)
- [x] All tests passing

### Validation
```bash
# Run tests
pytest redblackgraph/gpu/tests/test_avos_kernels.py -v

# Verify correctness
python -c "
from redblackgraph.gpu.kernels import avos_product
import cupy as cp
result = avos_product(cp.array([2, 3]), cp.array([3, 4]))
print(result)  # Should match reference
"
```

---

## Phase 2: Sparse Matrix Structure (Week 3)

### Goals
- Implement rb_matrix_gpu class
- Support creation from CPU matrices
- Implement CPU↔GPU transfers

### Tasks

#### 2.1 Implement rb_matrix_gpu Class

Create `redblackgraph/gpu/rb_matrix_gpu.py`:
```python
class rb_matrix_gpu:
    def __init__(self, matrix, device=0):
        # Transfer CPU→GPU
        
    def to_cpu(self):
        # Transfer GPU→CPU
        
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def nnz(self): ...
```

#### 2.2 Write Integration Tests

Create `redblackgraph/gpu/tests/test_gpu_matrix.py`:
```python
# Test:
# - Creation from various formats
# - CPU↔GPU roundtrip
# - Property access
# - Error handling
```

### Deliverables
- [x] rb_matrix_gpu class with CSR storage
- [x] CPU↔GPU transfer working
- [x] Integration tests passing

### Validation
```bash
pytest redblackgraph/gpu/tests/test_gpu_matrix.py -v

# Verify roundtrip
python -c "
from redblackgraph import rb_matrix
from redblackgraph.gpu import rb_matrix_gpu
import numpy as np

A = rb_matrix(np.eye(5, dtype=np.int32))
A_gpu = rb_matrix_gpu(A)
A_back = A_gpu.to_cpu()
assert (A != A_back).nnz == 0
print('Roundtrip successful')
"
```

---

## Phase 3: Sparse Matrix Multiplication (Weeks 4-5)

### Goals
- Implement two-pass CSR multiplication
- Full matrix multiplication working
- Correctness validated

### Tasks

#### 3.1 Implement Pass 1: Count Non-Zeros

In `redblackgraph/gpu/kernels/sparse_kernels.cu`:
```cuda
// Implement count_nnz_kernel
// - Hash table for tracking unique columns
// - Per-row parallel execution
// - Shared memory optimization
```

#### 3.2 Implement Pass 2: Compute Values

```cuda
// Implement compute_values_kernel
// - Hash table for accumulation
// - AVOS operations
// - Sort results by column index
```

#### 3.3 Create Sparse Matmul Wrapper

Create `redblackgraph/gpu/sparse_matmul.py`:
```python
def sparse_matmul_csr(A_data, A_indices, A_indptr,
                      B_data, B_indices, B_indptr, m, n):
    # Two-pass algorithm
    # Return (C_data, C_indices, C_indptr)
```

#### 3.4 Implement __matmul__ Operator

In `rb_matrix_gpu.py`:
```python
def __matmul__(self, other):
    # Validate inputs
    # Call sparse_matmul_csr
    # Return new rb_matrix_gpu
```

#### 3.5 Write Correctness Tests

Create `redblackgraph/gpu/tests/test_correctness.py`:
```python
# Property-based tests
# Random matrices vs CPU
# 1000+ test cases
```

### Deliverables
- [x] Working sparse matrix multiplication
- [x] @ operator functional
- [x] Bit-exact results matching CPU
- [x] All correctness tests passing

### Validation
```bash
# Run correctness tests
pytest redblackgraph/gpu/tests/test_correctness.py -v

# Manual verification
python -c "
from redblackgraph import rb_matrix
from redblackgraph.gpu import rb_matrix_gpu
import numpy as np

A = rb_matrix(np.random.randint(0, 10, (50, 50)))
A_gpu = rb_matrix_gpu(A)

C_cpu = A @ A
C_gpu = (A_gpu @ A_gpu).to_cpu()

assert (C_cpu != C_gpu).nnz == 0
print('Matrix multiplication correct!')
"
```

---

## Phase 4: Transitive Closure (Week 6)

### Goals
- Implement Warshall's algorithm on GPU
- Support transitive closure operation
- Validate on genealogy graphs

### Tasks

#### 4.1 Implement Transitive Closure

Create `redblackgraph/gpu/transitive_closure.py`:
```python
def transitive_closure_gpu(matrix):
    """Compute A* = A + A² + A³ + ... until convergence."""
    result = matrix
    prev_nnz = 0
    
    while result.nnz != prev_nnz:
        prev_nnz = result.nnz
        result = result @ matrix
        # Use AVOS sum to combine: result = avos_sum(result, new_power)
    
    return result
```

#### 4.2 Add to rb_matrix_gpu

```python
def transitive_closure(self):
    """Compute transitive closure."""
    from .transitive_closure import transitive_closure_gpu
    return transitive_closure_gpu(self)
```

#### 4.3 Test Transitive Closure

```python
# Test on genealogy graphs
# Verify against CPU implementation
# Check idempotency: (A*)* = A*
```

### Deliverables
- [x] Working transitive closure on GPU
- [x] Tests passing
- [x] Genealogy graphs validated

### Validation
```bash
# Test transitive closure
python -c "
from redblackgraph import rb_matrix, RED_ONE
from redblackgraph.gpu import rb_matrix_gpu
import numpy as np

# Small genealogy graph
A = rb_matrix(np.array([
    [RED_ONE, 2, 0],
    [0, RED_ONE, 3],
    [0, 0, BLACK_ONE]
]))

A_gpu = rb_matrix_gpu(A)
A_star_gpu = A_gpu.transitive_closure().to_cpu()
A_star_cpu = A.transitive_closure()

assert (A_star_cpu != A_star_gpu).nnz == 0
print('Transitive closure correct!')
"
```

---

## Phase 5: Performance Optimization (Weeks 7-8)

### Goals
- Profile and optimize kernels
- Achieve target speedups
- Document performance characteristics

### Tasks

#### 5.1 Baseline Performance

```bash
# Create benchmark suite
python -m redblackgraph.gpu.benchmarks.runner
```

#### 5.2 Profile with Nsight

```bash
# Profile sparse matmul
nsys profile --stats=true python benchmark_matmul.py

# Detailed kernel analysis
ncu --set full -o profile python benchmark_matmul.py
```

#### 5.3 Optimize Based on Profiling

Common optimizations:
- Tune block size (try 128, 256, 512)
- Adjust shared memory usage
- Optimize hash table size
- Reduce register pressure
- Minimize warp divergence

#### 5.4 Benchmark All Sizes

```python
# Run full benchmark suite
from redblackgraph.gpu.benchmarks import load_benchmark_matrices, BenchmarkRunner

matrices = load_benchmark_matrices()
runner = BenchmarkRunner(matrices)
results = runner.run_all()
runner.print_summary()
```

### Deliverables
- [x] Performance profiling completed
- [x] Optimizations implemented
- [x] Target speedups achieved (5-50x for large matrices)
- [x] Benchmark results documented

### Validation
- Large matrices (>5000×5000) show >10x speedup
- No performance regressions on small matrices
- Memory usage reasonable

---

## Phase 6: Documentation and Polish (Week 9)

### Goals
- Complete documentation
- User guide
- Examples
- API reference

### Tasks

#### 6.1 API Documentation

```python
# Add comprehensive docstrings
# Example in rb_matrix_gpu.py
class rb_matrix_gpu:
    """GPU-accelerated sparse Red-Black matrix.
    
    This class provides GPU acceleration for AVOS matrix operations
    using CUDA and CuPy. It maintains API compatibility with rb_matrix
    for easy migration.
    
    Examples:
        >>> # Transfer matrix to GPU
        >>> A_gpu = rb_matrix_gpu(A_cpu)
        >>> 
        >>> # Compute on GPU
        >>> C_gpu = A_gpu @ A_gpu
        >>> 
        >>> # Transfer back
        >>> C_cpu = C_gpu.to_cpu()
    
    See Also:
        redblackgraph.sparse.rb_matrix : CPU sparse matrix
    """
```

#### 6.2 User Guide

Create `docs/GPU_GUIDE.md`:
- Installation instructions
- Quick start examples
- Performance tips
- Troubleshooting
- FAQ

#### 6.3 Tutorial Notebook

Create `notebooks/GPU Tutorial.ipynb`:
- Basic usage
- Performance comparison
- Real genealogy examples
- Advanced features

### Deliverables
- [x] Complete API documentation
- [x] User guide written
- [x] Tutorial notebook created
- [x] Examples tested

---

## Phase 7: Integration and Testing (Week 10)

### Goals
- Run full test suite on GPU
- Ensure no regressions
- Prepare for merge

### Tasks

#### 7.1 Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run GPU-specific tests
pytest redblackgraph/gpu/tests/ -v

# Run with coverage
pytest redblackgraph/gpu/ --cov=redblackgraph.gpu --cov-report=html
```

#### 7.2 Integration Testing

```python
# Test GPU with existing workflows
# Verify backward compatibility
# Check for memory leaks
```

#### 7.3 Performance Regression Tests

```bash
# Compare against baseline
python scripts/check_performance_baseline.py
```

### Deliverables
- [x] All 167+ tests passing (including new GPU tests)
- [x] No performance regressions
- [x] Code coverage >85%
- [x] Integration validated

---

## Phase 8: Release Preparation (Weeks 11-12)

### Goals
- Prepare for release
- CI/CD setup
- Final polish

### Tasks

#### 8.1 CI/CD Configuration

Create `.github/workflows/gpu-ci.yml`:
```yaml
# Set up GPU runner
# Run tests on push/PR
# Generate coverage reports
# Upload artifacts
```

#### 8.2 Update pyproject.toml

```toml
[project.optional-dependencies]
gpu = [
    "cupy-cuda11x>=12.0.0; platform_machine == 'x86_64'",
    "cupy-cuda12x>=12.0.0; platform_machine == 'x86_64'",
]
```

#### 8.3 Update README

```markdown
## GPU Acceleration

RedBlackGraph now supports GPU acceleration for large sparse matrices:

```python
from redblackgraph.gpu import rb_matrix_gpu

# Transfer to GPU
A_gpu = rb_matrix_gpu(A)

# Compute on GPU (5-50x faster for large matrices)
result = (A_gpu @ A_gpu).to_cpu()
```

See [GPU Guide](docs/GPU_GUIDE.md) for details.
```

#### 8.4 Changelog

Update `CHANGELOG.md`:
```markdown
## [0.7.0] - GPU Support

### Added
- GPU acceleration via CuPy and CUDA
- `redblackgraph.gpu.rb_matrix_gpu` class
- Sparse CSR matrix multiplication on GPU
- Transitive closure on GPU
- Comprehensive GPU test suite
- Performance benchmarks

### Performance
- 5-50x speedup for matrices >1000×1000
- Bit-exact results matching CPU implementation
```

### Deliverables
- [x] CI/CD configured
- [x] Dependencies updated
- [x] Documentation complete
- [x] Release notes written
- [x] Ready to merge

---

## Success Criteria

### Functionality
- [ ] All AVOS operations working on GPU
- [ ] Sparse matrix multiplication correct
- [ ] Transitive closure working
- [ ] API compatible with rb_matrix

### Performance
- [ ] 5x+ speedup for 1000×1000 matrices
- [ ] 20x+ speedup for 10000×10000 matrices
- [ ] Memory usage reasonable

### Quality
- [ ] 100% test pass rate
- [ ] 85%+ code coverage
- [ ] No memory leaks
- [ ] Clean profiling results

### Documentation
- [ ] API fully documented
- [ ] User guide complete
- [ ] Examples working
- [ ] Tutorial notebook ready

---

## Risk Mitigation

### Risk: GPU Not Available in CI

**Mitigation**: 
- Make GPU tests optional
- Use `pytest.mark.skipif` for GPU tests
- Self-hosted runner for GPU CI

### Risk: Performance Not Meeting Targets

**Mitigation**:
- Profile early and often
- Start with correctness, optimize later
- Consider cuSPARSE integration if needed

### Risk: Memory Limitations

**Mitigation**:
- Implement out-of-core algorithms if needed
- Support batch processing
- Clear documentation of limits

### Risk: CuPy Compatibility Issues

**Mitigation**:
- Pin CuPy version requirements
- Test on multiple CUDA versions
- Provide fallback to CPU

---

## Progress Tracking

Create `progress.md` to track implementation:

```markdown
# GPU Implementation Progress

## Phase 0: Preparation ✅
- [x] Environment setup
- [x] Module structure
- [x] Test infrastructure

## Phase 1: Element-wise Operations ✅
- [x] AVOS sum kernel
- [x] AVOS product kernel
- [x] Unit tests

## Phase 2: Sparse Matrix Structure ✅
- [x] rb_matrix_gpu class
- [x] CPU↔GPU transfers
- [x] Integration tests

## Phase 3: Sparse Multiplication 🔄
- [x] Pass 1: Count non-zeros
- [ ] Pass 2: Compute values
- [ ] __matmul__ operator
- [ ] Correctness tests

## Phase 4: Transitive Closure ⏳
- [ ] Implementation
- [ ] Tests

## Phase 5: Optimization ⏳
- [ ] Profiling
- [ ] Optimizations
- [ ] Benchmarks

## Phase 6: Documentation ⏳
- [ ] API docs
- [ ] User guide
- [ ] Tutorial

## Phase 7: Integration ⏳
- [ ] Full test suite
- [ ] Regression tests

## Phase 8: Release ⏳
- [ ] CI/CD
- [ ] Release notes
- [ ] Merge
```

---

## Next Actions

1. **Review this plan** with team/stakeholders
2. **Set up GPU development environment** (Phase 0)
3. **Start with Phase 1**: Implement element-wise operations
4. **Track progress** in `progress.md`
5. **Weekly check-ins** to assess progress and adjust timeline

## Questions to Resolve

Before starting implementation:
- [ ] Which GPU hardware will be used for development?
- [ ] Is a GPU CI runner available?
- [ ] What CUDA version(s) need to be supported?
- [ ] Are there any specific performance requirements?
- [ ] Should we support multi-GPU from the start?

---

**Estimated completion**: 8-12 weeks from start  
**Prerequisites**: CUDA-capable GPU, CuPy installed, NumPy 2.x codebase  
**Dependencies**: Completion of NumPy 2.x migration (✅ Complete)
