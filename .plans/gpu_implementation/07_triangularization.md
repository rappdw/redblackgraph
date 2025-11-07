# GPU-Accelerated Triangularization & Preprocessing

## Overview

SpGEMM requires upper triangular input matrices. For Red-Black Graphs (DAGs), triangularization is achieved through **vertex reordering** (permutation), not structural filtering. This document outlines GPU acceleration strategies for this preprocessing step.

## Current State (Phase 1 Complete)

**CPU-Based Workflow:**
```python
# Method 1: Topological ordering (O(V+E), non-canonical)
from redblackgraph.reference.ordering import topological_ordering
ordering = topological_ordering(A_dag)
A_triangular = ordering.W

# Method 2: Canonical ordering (O(V²), deterministic)
from redblackgraph.sparse.csgraph import avos_canonical_ordering
ordering = avos_canonical_ordering(A_transitive_closure)
A_triangular = ordering.W

# Transfer to GPU
A_gpu = CSRMatrixGPU.from_cpu(A_triangular, triangular=True)
```

**Performance (10K vertices, 100K non-zeros):**
- Topological ordering: ~50-100 ms
- Canonical ordering: ~500-1000 ms
- Permutation application: ~50 ms
- GPU transfer (unified memory): ~10 ms

## Why Not `scipy.sparse.triu()`?

**Key Insight:** `triu()` performs **structural filtering** (deletes lower triangle), which **loses information** for DAGs/RBGs.

```python
# WRONG - loses edges
A_triangular = scipy.sparse.triu(A_dag)  # ❌ Deletes parent relationships

# CORRECT - preserves edges via reordering
ordering = topological_ordering(A_dag)
A_triangular = permute(A_dag, ordering.label_permutation)  # ✅ All edges preserved
```

**Example:**
```
Original DAG (child -> parent relationships):
       Alice Bob Carol
Alice  [ 1    0    0  ]  (self)
Bob    [ 2    1    0  ]  (Bob -> Alice relationship)
Carol  [ 0    3    1  ]  (Carol -> Bob relationship)

triu() result (WRONG):
[ 1  0  0 ]
[ 0  1  0 ]  ← Lost Bob -> Alice edge!
[ 0  0  1 ]  ← Lost Carol -> Bob edge!

Permutation result (CORRECT, reordered: Carol, Bob, Alice):
       Carol Bob Alice
Carol  [ 1    3    ?  ]
Bob    [ 0    1    2  ]
Alice  [ 0    0    1  ]  ← All relationships preserved
```

## GPU Acceleration Strategies

### Strategy 1: Hybrid (CPU Ordering + GPU Permutation)

**Approach:** Keep ordering on CPU, accelerate permutation on GPU

**Rationale:**
- Topological sort is inherently sequential (dependencies)
- Permutation application is embarrassingly parallel
- Unified memory eliminates transfer overhead

**Implementation:**
```python
def triangularize_hybrid(A_cpu: scipy.sparse.csr_matrix) -> CSRMatrixGPU:
    # Step 1: Compute ordering on CPU (O(V+E))
    ordering = topological_ordering(A_cpu)  # 50 ms
    perm = ordering.label_permutation
    
    # Step 2: Transfer to GPU
    A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=False)
    
    # Step 3: Apply permutation on GPU (O(nnz), parallel)
    A_tri_gpu = permute_gpu(A_gpu, perm)  # 10-15 ms
    
    return A_tri_gpu
```

**Expected Performance:**
- Total time: ~60-65 ms (vs 150 ms CPU-only)
- Speedup: 2-3x
- Implementation effort: Low (similar to SpGEMM structure)

**Components:**
1. `permute_symbolic_gpu()`: Count entries per row after permutation
2. `permute_numeric_gpu()`: Copy entries to new locations
3. Similar two-phase pattern as SpGEMM

### Strategy 2: Full GPU Pipeline

**Approach:** GPU topological sort + GPU permutation

**Rationale:**
- Maximum speedup potential
- Eliminates CPU bottleneck
- Enables end-to-end GPU pipeline

**Challenges:**
- GPU topological sort is complex (dependency coordination)
- Requires work queues and synchronization
- Not trivial to parallelize effectively

**Expected Performance:**
- Total time: ~15-30 ms
- Speedup: 5-10x vs CPU-only
- Implementation effort: Moderate-High

**Algorithm Sketch:**
```
1. Compute in-degrees on GPU (parallel count)
2. Queue vertices with in-degree = 0
3. Process queue levels in parallel:
   - Remove vertex from queue
   - Decrement neighbor in-degrees atomically
   - Add newly zero in-degree vertices to next level
4. Output: topological ordering
```

**References:**
- "Parallel Topological Sort" (Kahn's algorithm variant)
- CUB device-level algorithms for queue management

### Strategy 3: Lazy Evaluation (Future)

**Approach:** Don't materialize triangular matrix, apply mask during operations

**Rationale:**
- Avoid preprocessing entirely
- Trade memory for computation
- Good for one-shot operations

**Implementation:**
```python
class CSRMatrixGPU:
    def __init__(self, data, indices, indptr, shape, 
                 permutation=None):  # Store permutation lazily
        self.permutation = permutation
    
    def matmul(self, other):
        # Apply permutation implicitly during SpGEMM
        return spgemm_with_permutation(self, other, self.permutation)
```

## Recommended Implementation Plan

### Phase 3: Hybrid Triangularization (Medium Priority)

**Goals:**
- Implement GPU-accelerated permutation
- 2-3x speedup on permutation step
- Maintain exact CPU semantics

**Components:**
1. `redblackgraph/gpu/permutation_gpu.py` (~200 lines)
   - `permute_gpu(A_gpu, perm)` main API
   - `permute_symbolic_gpu()` kernel
   - `permute_numeric_gpu()` kernel
   - Tests validating against CPU `permute()`

2. `redblackgraph/gpu/triangularization.py` (~100 lines)
   - High-level API combining ordering + permutation
   - Support both topological and canonical methods
   - Auto-detection of already-triangular matrices

**Acceptance Criteria:**
- Bit-exact match with CPU permutation
- 2-3x faster than CPU on matrices >100K nnz
- All tests pass with CPU validation
- Works with both topological and canonical orderings

**Effort Estimate:** 4-6 hours

### Phase 4: Full GPU Pipeline (Lower Priority)

**Goals:**
- GPU topological sort implementation
- 5-10x end-to-end speedup
- Complete GPU-resident workflow

**Defer until:**
- Profiling shows ordering is >30% of total time
- Real workloads demonstrate need
- Phase 3 implementation complete

**Effort Estimate:** 2-3 days

## Performance Targets

Based on typical RBG workloads:

| Matrix Size | Method | Current (CPU) | Hybrid (Phase 3) | Full GPU (Phase 4) |
|-------------|--------|---------------|------------------|--------------------|
| 10K × 10K, 100K nnz | Topological | 150 ms | 60-75 ms | 15-30 ms |
| 100K × 100K, 1M nnz | Topological | 800 ms | 300-400 ms | 80-150 ms |
| 1M × 1M, 10M nnz | Topological | 5 sec | 2-2.5 sec | 500-800 ms |
| 10K × 10K, 100K nnz | Canonical | 1000 ms | 550-650 ms | N/A* |

*Canonical ordering is inherently O(V²), harder to parallelize effectively

## Integration with SpGEMM

**Current Usage:**
```python
# User responsible for triangularization
A_cpu = load_graph()
A_tri_cpu = topological_ordering(A_cpu).W
A_gpu = CSRMatrixGPU.from_cpu(A_tri_cpu, triangular=True)
C_gpu = spgemm_upper_triangular(A_gpu)
```

**Phase 3 Usage (Hybrid):**
```python
# GPU-accelerated triangularization
from redblackgraph.gpu import triangularize_gpu

A_cpu = load_graph()
A_tri_gpu = triangularize_gpu(A_cpu, method='topological')
C_gpu = spgemm_upper_triangular(A_tri_gpu)
```

**Phase 4 Usage (Full GPU):**
```python
# End-to-end GPU pipeline
A_gpu = CSRMatrixGPU.from_cpu(load_graph(), triangular=False)
A_tri_gpu = triangularize_gpu(A_gpu, method='topological')
C_gpu = spgemm_upper_triangular(A_tri_gpu)
```

## Testing Strategy

**Unit Tests:**
1. Small hand-crafted DAGs with known orderings
2. Validate permutation preserves all edges
3. Verify output is upper triangular
4. Bit-exact match with CPU reference

**Property Tests:**
1. Random DAGs of varying sizes
2. Check transitive closure invariant
3. Verify no information loss
4. Stress test with pathological cases (chains, stars, complete graphs)

**Performance Tests:**
1. Benchmark against CPU baseline
2. Measure speedup vs matrix size
3. Profile kernel execution times
4. Memory usage validation

## Open Questions

1. **Canonical ordering on GPU:** Is O(V²) component parallelizable enough to be worth it?
   - Likely NO for most cases
   - Keep canonical on CPU, only accelerate permutation

2. **Memory overhead:** How much temporary storage for permutation?
   - Symbolic: O(n) for new indptr
   - Numeric: O(nnz) for output
   - Total: 2× input size temporarily

3. **Integration point:** Where does triangularization fit in user workflow?
   - Explicit preprocessing step (Phase 3)
   - Implicit in SpGEMM (lazy evaluation, Phase 4+)
   - Both options with user control

## References

- Current CPU implementation: `redblackgraph/reference/ordering.py`
- Cython-optimized version: `redblackgraph/sparse/csgraph/_ordering.pyx`
- Permutation logic: `redblackgraph/sparse/csgraph/_permutation.pyx`
- GPU SpGEMM: `redblackgraph/gpu/spgemm*.py` (similar structure)

## Decision Log

**2025-11-06:** Documented triangularization strategies after Phase 1 completion
- Decision: Defer to Phase 3 based on profiling results
- Rationale: CPU topological sort is "fast enough" for current scale
- Revisit when: Real workloads show ordering >20% of total time
