# Performance Strategy and Benchmarking

## Performance Goals

### Target Speedups

Based on matrix size and sparsity:

| Matrix Size | Sparsity | Target Speedup | Notes |
|-------------|----------|----------------|-------|
| 100×100     | 1%       | 0.5-1x         | CPU faster (transfer overhead) |
| 500×500     | 1%       | 2-3x           | Break-even point |
| 1000×1000   | 1%       | 5-10x          | GPU starts winning |
| 5000×5000   | 1%       | 10-25x         | Sweet spot |
| 10000×10000 | 0.1%     | 20-50x         | Very sparse, ideal for GPU |
| 50000×50000 | 0.1%     | 30-100x        | Large scale |

### Realistic Expectations

**Good performance when**:
- Matrix size > 1000×1000
- Multiple operations on same data
- Batch processing
- Transitive closure on large graphs

**CPU may be faster when**:
- Matrix size < 500×500
- Single operation (transfer dominates)
- Very dense matrices (>10% dense)

## Profiling Strategy

### Tools

1. **NVIDIA Nsight Systems** - Overall timeline, kernel execution
2. **NVIDIA Nsight Compute** - Detailed kernel analysis
3. **CuPy profiler** - Python-level timing
4. **Custom timers** - End-to-end benchmarks

### Metrics to Track

#### Kernel-Level Metrics
- **Occupancy**: Target 50-75%
- **Memory bandwidth**: % of peak
- **Compute throughput**: FLOPS utilization
- **Warp efficiency**: Branch divergence
- **Register usage**: Per-thread registers
- **Shared memory**: Usage per block

#### System-Level Metrics
- **CPU→GPU transfer time**
- **GPU→CPU transfer time**
- **Kernel launch overhead**
- **Total execution time**
- **Memory allocation time**

## Optimization Priorities

### Phase 1: Correctness (Week 1-2)
- Focus: Make it work correctly
- No premature optimization
- Simple, clear implementations
- Comprehensive validation

### Phase 2: Basic Performance (Week 3-4)
- Profile and identify bottlenecks
- Optimize hot paths only
- Target low-hanging fruit:
  - Memory coalescing
  - Occupancy tuning
  - Block size optimization

### Phase 3: Advanced Optimization (Week 5+)
- Warp-level primitives
- Kernel fusion
- Custom memory allocators
- Multi-GPU support

## Benchmark Suite

### Test Matrices

```python
# redblackgraph/gpu/benchmarks/generate_test_matrices.py
import numpy as np
from scipy.sparse import random as sparse_random
from redblackgraph import rb_matrix

def generate_genealogy_like(n_vertices, avg_edges_per_vertex=3, seed=42):
    """Generate sparse matrix similar to genealogy graphs.
    
    Characteristics:
    - Very sparse (~0.1-1%)
    - Small out-degree (2-4 edges per vertex)
    - Integer values in genealogy-relevant range
    """
    np.random.seed(seed)
    
    density = avg_edges_per_vertex / n_vertices
    base = sparse_random(n_vertices, n_vertices, density=density,
                        format='csr', dtype=np.int32)
    
    # Replace data with genealogy-like values
    # Most values in range [2, 20], occasional -1 or 1
    base.data = np.random.randint(2, 20, size=base.data.size).astype(np.int32)
    
    # Add some identities
    n_identities = max(1, base.data.size // 100)
    identity_positions = np.random.choice(base.data.size, n_identities, replace=False)
    for i, pos in enumerate(identity_positions):
        base.data[pos] = -1 if i % 2 == 0 else 1
    
    return rb_matrix(base)

# Test suite
BENCHMARK_MATRICES = {
    'tiny': (100, 3),
    'small': (500, 3),
    'medium': (1000, 3),
    'large': (5000, 3),
    'xlarge': (10000, 2),
    'xxlarge': (50000, 2),
}

def load_benchmark_matrices():
    """Load all benchmark matrices."""
    matrices = {}
    for name, (size, edges) in BENCHMARK_MATRICES.items():
        matrices[name] = generate_genealogy_like(size, edges)
    return matrices
```

### Benchmark Runner

```python
# redblackgraph/gpu/benchmarks/runner.py
import time
import numpy as np
from redblackgraph import rb_matrix
from redblackgraph.gpu import rb_matrix_gpu
import cupy as cp

class BenchmarkRunner:
    """Run and collect GPU benchmarks."""
    
    def __init__(self, matrices):
        self.matrices = matrices
        self.results = []
        
    def benchmark_matmul(self, matrix_name, n_runs=10):
        """Benchmark matrix multiplication A @ A.
        
        Measures:
        - CPU time
        - GPU time (including transfers)
        - GPU time (no transfers)
        - Speedup
        """
        A_cpu = self.matrices[matrix_name]
        
        # Warm-up
        _ = A_cpu @ A_cpu
        
        # CPU benchmark
        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            C_cpu = A_cpu @ A_cpu
            cpu_times.append(time.perf_counter() - start)
        
        cpu_mean = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)
        
        # GPU benchmark (with transfers)
        gpu_with_transfer_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            A_gpu = rb_matrix_gpu(A_cpu)
            C_gpu = A_gpu @ A_gpu
            C_cpu_result = C_gpu.to_cpu()
            gpu_with_transfer_times.append(time.perf_counter() - start)
        
        gpu_with_mean = np.mean(gpu_with_transfer_times)
        
        # GPU benchmark (no transfers)
        A_gpu = rb_matrix_gpu(A_cpu)
        cp.cuda.Stream.null.synchronize()  # Ensure transfer complete
        
        gpu_no_transfer_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            C_gpu = A_gpu @ A_gpu
            cp.cuda.Stream.null.synchronize()
            gpu_no_transfer_times.append(time.perf_counter() - start)
        
        gpu_no_mean = np.mean(gpu_no_transfer_times)
        gpu_no_std = np.std(gpu_no_transfer_times)
        
        # Calculate speedups
        speedup_with_transfer = cpu_mean / gpu_with_mean
        speedup_no_transfer = cpu_mean / gpu_no_mean
        
        result = {
            'matrix': matrix_name,
            'shape': A_cpu.shape,
            'nnz': A_cpu.nnz,
            'density': A_cpu.nnz / (A_cpu.shape[0] * A_cpu.shape[1]),
            'cpu_time': cpu_mean,
            'cpu_std': cpu_std,
            'gpu_time_with_transfer': gpu_with_mean,
            'gpu_time_no_transfer': gpu_no_mean,
            'gpu_std': gpu_no_std,
            'speedup_with_transfer': speedup_with_transfer,
            'speedup_no_transfer': speedup_no_transfer,
        }
        
        self.results.append(result)
        return result
    
    def run_all(self):
        """Run all benchmarks."""
        print("Running GPU Benchmarks")
        print("=" * 80)
        
        for name in self.matrices:
            print(f"\nBenchmarking {name}...", end=" ")
            result = self.benchmark_matmul(name)
            print(f"Speedup: {result['speedup_no_transfer']:.2f}x")
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary table."""
        print("\n\nBenchmark Summary")
        print("=" * 120)
        print(f"{'Matrix':<12} {'Shape':<15} {'NNZ':<10} {'CPU (ms)':<12} "
              f"{'GPU (ms)':<12} {'Speedup':<10}")
        print("-" * 120)
        
        for r in self.results:
            shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
            print(f"{r['matrix']:<12} {shape_str:<15} {r['nnz']:<10} "
                  f"{r['cpu_time']*1000:>11.2f} {r['gpu_time_no_transfer']*1000:>11.2f} "
                  f"{r['speedup_no_transfer']:>9.2f}x")
```

## Memory Optimization

### Memory Usage Estimation

```python
def estimate_gpu_memory(matrix_shape, nnz, dtype=np.int32):
    """Estimate GPU memory needed for matrix.
    
    Args:
        matrix_shape: (m, n)
        nnz: Number of non-zeros
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    m, n = matrix_shape
    dtype_size = np.dtype(dtype).itemsize
    
    # CSR storage
    data_size = nnz * dtype_size
    indices_size = nnz * 4  # int32
    indptr_size = (m + 1) * 4  # int32
    
    # Working memory for matmul (worst case: output has same nnz)
    output_size = data_size + indices_size + indptr_size
    
    # Hash tables in shared memory (per-warp)
    # Not included here as it's per-SM, not global
    
    total = data_size + indices_size + indptr_size + output_size
    
    return {
        'input_matrix': data_size + indices_size + indptr_size,
        'output_matrix': output_size,
        'total': total,
        'total_mb': total / (1024**2),
    }
```

### Memory Transfer Optimization

```python
# Strategies to minimize transfers

# 1. Batch operations
matrices = [rb_matrix_gpu(m) for m in cpu_matrices]  # Transfer once
results = [m @ m for m in matrices]  # Compute on GPU
cpu_results = [r.to_cpu() for r in results]  # Transfer back once

# 2. Keep hot data on GPU
class GPUCache:
    """Cache frequently used matrices on GPU."""
    
    def __init__(self, max_size_mb=1024):
        self.cache = {}
        self.max_size = max_size_mb * 1024**2
        self.current_size = 0
    
    def get(self, key, matrix_cpu):
        """Get GPU matrix from cache or transfer."""
        if key not in self.cache:
            self.cache[key] = rb_matrix_gpu(matrix_cpu)
            self.current_size += estimate_size(matrix_cpu)
        return self.cache[key]
```

## Kernel Optimization Techniques

### 1. Shared Memory Optimization

```cuda
// Good: Efficient shared memory usage
__shared__ int col_hash[WARP_SIZE * HASH_SIZE];
int* my_hash = &col_hash[warp_id * HASH_SIZE];

// Bad: Bank conflicts
__shared__ int data[256];  // 32 banks, stride 32 causes conflicts
int val = data[threadIdx.x];  // Potential conflict
```

### 2. Warp Divergence Minimization

```cuda
// Good: Minimal divergence
if (warp_any_sync(0xffffffff, condition)) {
    // All threads in warp execute
}

// Bad: High divergence
if (threadIdx.x % 2 == 0) {  // Half of threads diverge
    // Heavy computation
}
```

### 3. Memory Coalescing

```cuda
// Good: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
T val = global_array[idx];

// Bad: Strided access
int idx = threadIdx.x * stride;
T val = global_array[idx];
```

## Performance Testing

### Unit Performance Tests

```python
# redblackgraph/gpu/tests/test_performance.py
import pytest
import numpy as np
from redblackgraph.gpu import rb_matrix_gpu
from .benchmarks.generate_test_matrices import generate_genealogy_like

@pytest.mark.benchmark
def test_medium_matrix_performance():
    """Test that GPU is faster than CPU for medium matrices."""
    A = generate_genealogy_like(2000, avg_edges_per_vertex=3)
    
    # CPU baseline
    cpu_result = A @ A
    
    # GPU timing
    A_gpu = rb_matrix_gpu(A)
    gpu_result = (A_gpu @ A_gpu).to_cpu()
    
    # Verify correctness
    assert (cpu_result != gpu_result).nnz == 0, "Results don't match"
    
    # This is a smoke test; detailed benchmarking done separately
    print(f"Matrix size: {A.shape}, nnz: {A.nnz}")

@pytest.mark.benchmark
@pytest.mark.parametrize("size", [1000, 5000, 10000])
def test_scaling_performance(size):
    """Test performance scaling with matrix size."""
    A = generate_genealogy_like(size, avg_edges_per_vertex=3)
    A_gpu = rb_matrix_gpu(A)
    result = (A_gpu @ A_gpu).to_cpu()
    
    # Just verify it works; timing collected separately
    assert result.shape == A.shape
```

## Continuous Performance Monitoring

### Integration with CI

```yaml
# .github/workflows/gpu-benchmark.yml
name: GPU Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run benchmarks
      run: |
        python -m redblackgraph.gpu.benchmarks.runner
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results.json
    
    - name: Check for regressions
      run: |
        python scripts/check_performance_regression.py
```

### Performance Dashboard

Track performance over time:
- Speedup by matrix size
- Memory usage trends
- Kernel occupancy
- Regression detection

## Optimization Checklist

Before declaring performance work "done":

- [ ] Profile with Nsight Systems
- [ ] Achieve >50% occupancy on key kernels
- [ ] Memory bandwidth >60% of peak for memory-bound kernels
- [ ] Speedup >5x for matrices >1000×1000
- [ ] No performance regressions vs baseline
- [ ] Documentation of optimization decisions
- [ ] Benchmarks added to test suite

## Next Steps

Read **[05_testing_plan.md](05_testing_plan.md)** for comprehensive testing strategy.
