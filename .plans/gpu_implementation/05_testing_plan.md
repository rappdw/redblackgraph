# Testing Plan

## Testing Philosophy

**Golden Rule**: GPU implementation must produce **bit-exact** results matching CPU implementation.

No approximations, no tolerance for numerical differences. The AVOS algebra is defined over integers, and integer operations on GPU must match CPU exactly.

## Test Hierarchy

### Level 1: Unit Tests (CUDA Kernels)
Test individual CUDA kernels in isolation.

### Level 2: Integration Tests (Python API)
Test Python wrapper and CuPy integration.

### Level 3: Correctness Tests (CPU Comparison)
Verify GPU results match CPU reference implementation.

### Level 4: Regression Tests (Existing Test Suite)
Run all existing tests with GPU backend.

### Level 5: Performance Tests (Benchmarks)
Validate performance improvements.

## Level 1: CUDA Kernel Unit Tests

### Test: AVOS Sum Kernel

```python
# redblackgraph/gpu/tests/test_avos_kernels.py
import pytest
import numpy as np
import cupy as cp
from redblackgraph.gpu.kernels import avos_sum

@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_avos_sum_basic(dtype):
    """Test basic AVOS sum properties."""
    x = cp.array([0, 3, 5, 0, 10], dtype=dtype)
    y = cp.array([7, 0, 3, 0, 8], dtype=dtype)
    
    result = avos_sum(x, y)
    expected = cp.array([7, 3, 3, 0, 8], dtype=dtype)
    
    assert cp.all(result == expected)

def test_avos_sum_zero_identity():
    """Test that 0 is identity for avos_sum."""
    x = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    zero = cp.zeros_like(x)
    
    assert cp.all(avos_sum(x, zero) == x)
    assert cp.all(avos_sum(zero, x) == x)

def test_avos_sum_commutative():
    """Test commutativity: avos_sum(x, y) == avos_sum(y, x)."""
    x = cp.random.randint(0, 100, size=1000, dtype=np.int32)
    y = cp.random.randint(0, 100, size=1000, dtype=np.int32)
    
    assert cp.all(avos_sum(x, y) == avos_sum(y, x))

def test_avos_sum_signed_values():
    """Test handling of signed values (RED_ONE, BLACK_ONE)."""
    x = cp.array([-1, 1, -1, 1, 0], dtype=np.int32)
    y = cp.array([5, 0, 0, 5, -1], dtype=np.int32)
    
    # -1 should be treated as largest unsigned value
    result = avos_sum(x, y)
    
    # Verify against reference implementation
    from redblackgraph.reference.rbg_math import avos_sum as ref_sum
    expected = cp.array([ref_sum(int(x[i]), int(y[i])) 
                        for i in range(len(x))], dtype=np.int32)
    
    assert cp.all(result == expected)
```

### Test: AVOS Product Kernel

```python
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_avos_product_basic(dtype):
    """Test basic AVOS product."""
    from redblackgraph.gpu.kernels import avos_product
    
    x = cp.array([2, 3, 0, 5], dtype=dtype)
    y = cp.array([3, 4, 5, 0], dtype=dtype)
    
    result = avos_product(x, y)
    
    # Verify against reference
    from redblackgraph.reference.rbg_math import avos_product as ref_prod
    expected = cp.array([ref_prod(int(x[i]), int(y[i])) 
                        for i in range(len(x))], dtype=dtype)
    
    assert cp.all(result == expected)

def test_avos_product_zero_annihilator():
    """Test that 0 is annihilator for avos_product."""
    x = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    zero = cp.zeros_like(x)
    
    assert cp.all(avos_product(x, zero) == 0)
    assert cp.all(avos_product(zero, x) == 0)

def test_avos_product_identities_special_cases():
    """Test RED_ONE and BLACK_ONE special cases."""
    from redblackgraph.gpu.kernels import avos_product
    from redblackgraph import RED_ONE, BLACK_ONE
    
    # Same-gender self-loops
    assert int(avos_product(cp.array([RED_ONE]), cp.array([RED_ONE]))[0]) == RED_ONE
    assert int(avos_product(cp.array([BLACK_ONE]), cp.array([BLACK_ONE]))[0]) == BLACK_ONE
    
    # Cross-gender undefined
    assert int(avos_product(cp.array([RED_ONE]), cp.array([BLACK_ONE]))[0]) == 0
    assert int(avos_product(cp.array([BLACK_ONE]), cp.array([RED_ONE]))[0]) == 0

def test_avos_product_parity_constraints_right():
    """Test parity constraints when identity on RIGHT."""
    from redblackgraph.gpu.kernels import avos_product
    from redblackgraph import RED_ONE, BLACK_ONE
    
    # RED_ONE on right: even passes, odd filtered
    even_vals = cp.array([2, 4, 6, 8], dtype=np.int32)
    odd_vals = cp.array([3, 5, 7, 9], dtype=np.int32)
    red_one = cp.full_like(even_vals, RED_ONE)
    
    result_even = avos_product(even_vals, red_one)
    result_odd = avos_product(odd_vals, red_one)
    
    assert cp.all(result_even == even_vals)  # Even passes through
    assert cp.all(result_odd == 0)  # Odd filtered to 0
    
    # BLACK_ONE on right: odd passes, even filtered
    black_one = cp.full_like(even_vals, BLACK_ONE)
    
    result_even = avos_product(even_vals, black_one)
    result_odd = avos_product(odd_vals, black_one)
    
    assert cp.all(result_even == 0)  # Even filtered to 0
    assert cp.all(result_odd == odd_vals)  # Odd passes through

def test_avos_product_parity_constraints_left():
    """Test that identity on LEFT acts as starting point (no filtering)."""
    from redblackgraph.gpu.kernels import avos_product
    from redblackgraph import RED_ONE, BLACK_ONE
    
    values = cp.array([2, 3, 4, 5], dtype=np.int32)
    
    # RED_ONE on left: treated as 1, normal bit-shift
    result_red = avos_product(cp.full_like(values, RED_ONE), values)
    
    # BLACK_ONE on left: treated as 1, normal bit-shift  
    result_black = avos_product(cp.full_like(values, BLACK_ONE), values)
    
    # Verify against reference
    from redblackgraph.reference.rbg_math import avos_product as ref_prod
    expected_red = cp.array([ref_prod(RED_ONE, int(v)) for v in values], dtype=np.int32)
    expected_black = cp.array([ref_prod(BLACK_ONE, int(v)) for v in values], dtype=np.int32)
    
    assert cp.all(result_red == expected_red)
    assert cp.all(result_black == expected_black)

@pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
def test_avos_product_random_vs_reference(size):
    """Test random inputs against reference implementation."""
    from redblackgraph.gpu.kernels import avos_product
    from redblackgraph.reference.rbg_math import avos_product as ref_prod
    
    # Generate random values including identities
    x_np = np.random.randint(-1, 20, size=size, dtype=np.int32)
    y_np = np.random.randint(-1, 20, size=size, dtype=np.int32)
    
    # Replace some values with identities
    identity_mask = np.random.random(size) < 0.1
    x_np[identity_mask] = np.random.choice([-1, 1], size=identity_mask.sum())
    y_np[identity_mask] = np.random.choice([-1, 1], size=identity_mask.sum())
    
    x = cp.array(x_np)
    y = cp.array(y_np)
    
    result = avos_product(x, y)
    expected = np.array([ref_prod(int(x_np[i]), int(y_np[i])) 
                        for i in range(size)], dtype=np.int32)
    
    assert cp.all(result == cp.array(expected))
```

## Level 2: Integration Tests

### Test: rb_matrix_gpu Class

```python
# redblackgraph/gpu/tests/test_gpu_matrix.py
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from redblackgraph import rb_matrix, RED_ONE, BLACK_ONE
from redblackgraph.gpu import rb_matrix_gpu

def test_create_from_rb_matrix():
    """Test creating GPU matrix from CPU rb_matrix."""
    A_cpu = rb_matrix(csr_matrix([[RED_ONE, 2, 0],
                                   [0, BLACK_ONE, 3],
                                   [4, 0, 5]]))
    A_gpu = rb_matrix_gpu(A_cpu)
    
    assert A_gpu.shape == A_cpu.shape
    assert A_gpu.dtype == A_cpu.dtype
    assert A_gpu.nnz == A_cpu.nnz

def test_to_cpu_roundtrip():
    """Test CPU→GPU→CPU roundtrip preserves data."""
    A_cpu = rb_matrix(csr_matrix(np.random.randint(0, 10, size=(10, 10))))
    A_gpu = rb_matrix_gpu(A_cpu)
    A_back = A_gpu.to_cpu()
    
    assert (A_cpu != A_back).nnz == 0

def test_matmul_small():
    """Test small matrix multiplication."""
    A = rb_matrix(csr_matrix([[RED_ONE, 2],
                               [3, BLACK_ONE]]))
    
    A_gpu = rb_matrix_gpu(A)
    result_gpu = (A_gpu @ A_gpu).to_cpu()
    result_cpu = A @ A
    
    assert (result_cpu != result_gpu).nnz == 0

def test_matmul_medium():
    """Test medium-sized matrix multiplication."""
    np.random.seed(42)
    A = rb_matrix(csr_matrix(np.random.randint(0, 10, size=(100, 100))))
    
    A_gpu = rb_matrix_gpu(A)
    result_gpu = (A_gpu @ A_gpu).to_cpu()
    result_cpu = A @ A
    
    assert (result_cpu != result_gpu).nnz == 0

def test_shape_validation():
    """Test that incompatible shapes raise error."""
    A = rb_matrix_gpu(rb_matrix(csr_matrix(np.eye(5))))
    B = rb_matrix_gpu(rb_matrix(csr_matrix(np.eye(3))))
    
    with pytest.raises(ValueError, match="Incompatible shapes"):
        _ = A @ B

def test_dtype_validation():
    """Test that dtype mismatch raises error."""
    A = rb_matrix_gpu(rb_matrix(csr_matrix(np.eye(5, dtype=np.int32))))
    B = rb_matrix_gpu(rb_matrix(csr_matrix(np.eye(5, dtype=np.int64))))
    
    with pytest.raises(ValueError, match="Dtype mismatch"):
        _ = A @ B
```

## Level 3: Correctness Tests

### Strategy: Property-Based Testing

```python
# redblackgraph/gpu/tests/test_correctness.py
import pytest
import numpy as np
from hypothesis import given, strategies as st
from scipy.sparse import random as sparse_random
from redblackgraph import rb_matrix
from redblackgraph.gpu import rb_matrix_gpu

@given(
    size=st.integers(min_value=5, max_value=100),
    density=st.floats(min_value=0.01, max_value=0.2),
    seed=st.integers(min_value=0, max_value=1000)
)
def test_matmul_matches_cpu(size, density, seed):
    """Property test: GPU matmul matches CPU for random matrices."""
    np.random.seed(seed)
    
    A_sparse = sparse_random(size, size, density=density, 
                            format='csr', dtype=np.int32)
    A_sparse.data = np.random.randint(-1, 20, size=A_sparse.data.size).astype(np.int32)
    
    A_cpu = rb_matrix(A_sparse)
    A_gpu = rb_matrix_gpu(A_cpu)
    
    result_cpu = A_cpu @ A_cpu
    result_gpu = (A_gpu @ A_gpu).to_cpu()
    
    # Bit-exact comparison
    assert (result_cpu != result_gpu).nnz == 0, \
        f"GPU result differs from CPU (size={size}, density={density}, seed={seed})"

def test_transitive_closure_correctness():
    """Test transitive closure matches CPU implementation."""
    # Create test genealogy graph
    A_np = np.array([
        [RED_ONE, 2, 3, 0, 0],
        [0, RED_ONE, 0, 2, 0],
        [0, 0, BLACK_ONE, 0, 3],
        [0, 0, 0, RED_ONE, 0],
        [0, 0, 0, 0, BLACK_ONE]
    ], dtype=np.int32)
    
    A_cpu = rb_matrix(csr_matrix(A_np))
    A_gpu = rb_matrix_gpu(A_cpu)
    
    # Compute transitive closure
    closure_cpu = A_cpu.transitive_closure()
    closure_gpu = A_gpu.transitive_closure().to_cpu()
    
    assert (closure_cpu != closure_gpu).nnz == 0
```

## Level 4: Regression Tests

### Run Existing Test Suite on GPU

```python
# redblackgraph/gpu/tests/test_regression.py
"""Run existing CPU tests on GPU implementation.

This ensures GPU implementation passes all existing tests.
"""
import pytest
from redblackgraph import rb_matrix
from redblackgraph.gpu import rb_matrix_gpu

# Import existing test modules
from tests.sparse import test_sparse_matmul
from tests.avos import test_relational_composition

class TestGPURegression:
    """Run CPU tests on GPU implementation."""
    
    @pytest.fixture
    def matrix_class(self):
        """Override matrix class to use GPU."""
        return rb_matrix_gpu
    
    # Import and adapt CPU tests
    # (This is a pattern; actual implementation would use test parameterization)
    
    def test_sparse_matmul_basic(self):
        """Test basic sparse matmul from existing tests."""
        # Reuse logic from test_sparse_matmul.test_basic
        # but with GPU matrices
        pass

# Alternative: Parameterize CPU tests to run on GPU
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_matmul_backends(backend):
    """Test that both backends produce same results."""
    A_cpu = rb_matrix(...)
    
    if backend == "gpu":
        A = rb_matrix_gpu(A_cpu)
    else:
        A = A_cpu
    
    result = A @ A
    
    if backend == "gpu":
        result = result.to_cpu()
    
    # Verify against reference
    assert ...
```

## Level 5: Performance Tests

See **[04_performance_strategy.md](04_performance_strategy.md)** for details.

## Edge Cases to Test

### Special Values
- [ ] Zero matrices
- [ ] Identity matrices  
- [ ] Matrices with only RED_ONE
- [ ] Matrices with only BLACK_ONE
- [ ] Mixed identity values
- [ ] Negative values (invalid, should error)

### Shapes
- [ ] Square matrices
- [ ] Tall matrices (m >> n)
- [ ] Wide matrices (m << n)
- [ ] Single row/column
- [ ] Empty matrices

### Sparsity
- [ ] Very sparse (0.01% density)
- [ ] Medium sparse (1% density)
- [ ] Dense sparse (10% density)
- [ ] Single non-zero
- [ ] All non-zeros on diagonal

### Dtypes
- [ ] int8 (small values, overflow risk)
- [ ] int16
- [ ] int32 (most common)
- [ ] int64 (large values)

## Test Organization

```
redblackgraph/gpu/tests/
├── __init__.py
├── test_avos_kernels.py       # Level 1: Kernel unit tests
├── test_gpu_matrix.py          # Level 2: Integration tests
├── test_correctness.py         # Level 3: CPU comparison
├── test_regression.py          # Level 4: Existing tests
├── test_performance.py         # Level 5: Performance tests
├── test_edge_cases.py          # Edge cases
├── conftest.py                 # Pytest configuration
└── fixtures/
    ├── test_matrices.py        # Reusable test matrices
    └── reference_results.py    # Known-good results
```

## Continuous Integration

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests

on: [push, pull_request]

jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        cuda-version: ['11.8', '12.1']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
        pip install cupy-cuda${{ matrix.cuda-version }}
    
    - name: Run GPU tests
      run: |
        pytest redblackgraph/gpu/tests/ -v --tb=short
    
    - name: Run CPU comparison tests
      run: |
        pytest redblackgraph/gpu/tests/test_correctness.py -v
```

## Test Coverage Goals

- **Unit tests**: >90% line coverage of kernel wrappers
- **Integration tests**: All public API methods tested
- **Correctness tests**: 1000+ random matrices vs CPU
- **Regression tests**: 100% of existing tests pass on GPU
- **Performance tests**: All benchmark matrices tested

## Debugging Failed Tests

### Common Issues

1. **Bit-exact mismatch**: Check for race conditions, atomic operations
2. **Memory errors**: Check array bounds, shared memory size
3. **Launch configuration**: Verify block/thread counts
4. **Dtype issues**: Ensure proper type casting

### Debug Tools

```python
# Enable CUDA error checking
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# CuPy debug mode
import cupy as cp
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# Print from kernel (use sparingly)
# In CUDA: printf("Debug: %d\n", value);
```

## Test Execution

```bash
# Run all GPU tests
pytest redblackgraph/gpu/tests/ -v

# Run specific test level
pytest redblackgraph/gpu/tests/test_avos_kernels.py -v

# Run with coverage
pytest redblackgraph/gpu/tests/ --cov=redblackgraph.gpu --cov-report=html

# Run performance tests only
pytest redblackgraph/gpu/tests/test_performance.py -m benchmark

# Run correctness tests with hypothesis
pytest redblackgraph/gpu/tests/test_correctness.py --hypothesis-show-statistics
```

## Next Steps

Read **[06_implementation_phases.md](06_implementation_phases.md)** for step-by-step implementation roadmap.
