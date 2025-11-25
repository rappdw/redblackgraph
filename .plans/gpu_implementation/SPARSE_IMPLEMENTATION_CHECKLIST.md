# Sparse Matrix Implementation - Execution Checklist

**Goal:** Enable canonical form computation on 100K+ vertex graphs using truly sparse operations.

**Timeline:** 17-25 days total

---

## Phase 0: Sparse Infrastructure (CRITICAL FOUNDATION)
**Duration:** 4-6 days | **Must complete before all other phases**

### 0.1: CSR Iteration Primitives
**File:** `redblackgraph/sparse/csgraph/_csr_utils.pxi`

- [ ] Create `_csr_utils.pxi` file
- [ ] Implement inline row edge iterator macro
  ```cython
  cdef inline iterate_row_edges(ITYPE_t[:] indptr, ITYPE_t[:] indices, DTYPE_t[:] data, int vertex)
  ```
- [ ] Implement CSRRowIterator class for Python-accessible iteration
- [ ] Add column edge iterator (requires CSC format)
- [ ] Document usage patterns

**Test:** `tests/sparse/test_csr_utils.py`
- [ ] Create test file
- [ ] Test row iteration matches manual index access
- [ ] Test iteration only visits non-zero entries
- [ ] Benchmark: Verify O(nnz) not O(n²)

---

### 0.2: Sparse Matrix Permutation
**File:** `redblackgraph/sparse/csgraph/_permutation.pyx`

- [ ] Implement `permute_sparse(A_csr, p)` function
  - Algorithm:
    1. Create inverse permutation p_inv
    2. Count edges per output row
    3. Build output indptr from counts
    4. Iterate input rows in permuted order
    5. Remap column indices via p_inv
    6. Copy to output arrays
- [ ] Handle upper triangular optimization flag
- [ ] Update existing `permute()` to dispatch to sparse version
- [ ] Preserve CSR format (no densification!)

**Test:** `tests/sparse/test_sparse_permutation.py`
- [ ] Create test file
- [ ] Test permutation preserves graph structure
- [ ] Test NO densification occurs (check output.nnz == input.nnz)
- [ ] Compare with dense permutation results
- [ ] Test upper triangular case

---

### 0.3: Format Conversion Utilities
**File:** `redblackgraph/sparse/csgraph/_sparse_format.py`

- [ ] Create `_sparse_format.py` file
- [ ] Implement `ensure_csr(A)` - convert to CSR if needed
- [ ] Implement `ensure_csc(A)` - convert to CSC for column access
- [ ] Implement `to_dense_if_needed(A, threshold)` - conditional densification
- [ ] Implement `get_density(A)` helper
- [ ] Add format detection utilities

**Test:** `tests/sparse/test_format_conversion.py`
- [ ] Create test file
- [ ] Test CSR/CSC round-trip
- [ ] Test density threshold logic
- [ ] Test format detection

---

### 0.4: Transpose Utilities
**File:** `redblackgraph/sparse/csgraph/_csr_utils.pxi`

- [ ] Implement `build_csr_transpose(A_csr)` in Cython
  - O(nnz) single-pass algorithm
  - Build CSR format of A^T
- [ ] Implement caching mechanism for repeated transpose
- [ ] Add CSR → CSC conversion helper

**Test:** `tests/sparse/test_transpose.py`
- [ ] Create test file
- [ ] Test transpose correctness
- [ ] Test O(nnz) performance
- [ ] Test cache effectiveness

---

### 0.5: Density Monitoring
**File:** `redblackgraph/sparse/csgraph/_density.py`

- [ ] Create `_density.py` file
- [ ] Implement `DensityMonitor` class
  - `__init__(warn_threshold, error_threshold)`
  - `check(A, operation_name)` - monitor and warn/error
  - `get_density(A)` - compute current density
  - `history` - track density through pipeline
- [ ] Implement `DensificationError` exception
- [ ] Implement `DensificationWarning` warning class

**Test:** `tests/sparse/test_density_monitoring.py`
- [ ] Create test file
- [ ] Test warning triggers
- [ ] Test error triggers
- [ ] Test history tracking

---

### 0.6: Component Extraction & Merge
**File:** `redblackgraph/sparse/csgraph/_components.pyx`

- [ ] Implement `extract_submatrix(A_csr, vertices)` - sparse extraction
  - Build mapping: old_id → new_id
  - Extract rows for specified vertices
  - Remap column indices
  - Return sparse submatrix
- [ ] Implement `merge_component_matrices(components, n_total)`
  - Pre-allocate full matrix structure
  - Place each component at correct offset
  - Return full sparse matrix

**Test:** `tests/sparse/test_component_extraction.py`
- [ ] Create test file
- [ ] Test extraction preserves subgraph
- [ ] Test merge reconstructs full graph
- [ ] Test round-trip: extract → merge → compare

---

### 0.7: Build System Updates
**Files:** `redblackgraph/sparse/csgraph/meson.build`, `__init__.py`

- [ ] Add `_csr_utils.pxi` to meson.build includes
- [ ] Add `_sparse_format.py` to Python sources
- [ ] Add `_density.py` to Python sources
- [ ] Update `__init__.py` to export new utilities
- [ ] Test build on clean environment

---

## Phase 1: Sparse Topological Sort
**Duration:** 1-2 days | **Depends on:** Phase 0

### 1.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_topological_sort.pyx`

- [ ] Create `_topological_sort.pyx` file
- [ ] Implement `topological_sort(A)` function
  - Use Phase 0 CSR iteration primitives
  - Iterative DFS (stack-based, no recursion)
  - Handle both CSR and rb_array inputs
  - Return permutation array
- [ ] Implement `topological_ordering(A)` wrapper
  - Call topological_sort
  - Use Phase 0 sparse permutation
  - Return Ordering object
- [ ] Add to `__init__.py` exports

**Test:** `tests/sparse/test_topological_sort.py`
- [ ] Create test file
- [ ] Compare output with reference implementation
- [ ] Test on sparse CSR matrices (100K vertices)
- [ ] Verify upper triangular property of result
- [ ] Test with multiple disconnected components
- [ ] Benchmark: Confirm O(V+E) scaling

---

### 1.2: Build Integration
- [ ] Add to `meson.build`
- [ ] Update package `__init__.py`
- [ ] Rebuild and test

---

## Phase 2a: Component-Wise Processing
**Duration:** 2-3 days | **Depends on:** Phase 0, Phase 3

### 2a.1: Implementation
**File:** `redblackgraph/sparse/csgraph/transitive_closure.py`

- [ ] Implement `component_wise_closure(A, components, method)`
  - Use Phase 0 extraction to get submatrices
  - Compute closure per component (can densify small matrices)
  - Use Phase 0 merge to reconstruct full matrix
  - Return sparse result (sparse between components)

**Test:** `tests/sparse/test_component_closure.py`
- [ ] Create test file
- [ ] Test on multi-component graph
- [ ] Verify result equals full closure
- [ ] Measure memory savings vs full graph
- [ ] Test parallelization potential (mark for future)

---

## Phase 2b: Upper Triangular Floyd-Warshall
**Duration:** 2-3 days | **Depends on:** Phase 1

### 2b.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_shortest_path.pyx`

- [ ] Modify `_floyd_warshall_avos()` to accept `assume_upper_triangular` flag
- [ ] Implement optimized triple loop:
  ```cython
  for k in range(N):
      for i in range(k+1):      # Only i ≤ k
          for j in range(k, N):  # Only j ≥ k
  ```
- [ ] Update `floyd_warshall()` wrapper to accept parameter
- [ ] Update `shortest_path()` to pass through parameter
- [ ] Update `transitive_closure_floyd_warshall()` wrapper

**Test:** `tests/sparse/test_upper_triangular_closure.py`
- [ ] Create test file
- [ ] Test correctness on upper triangular matrices
- [ ] Measure speedup (expect 1.8-2x)
- [ ] Verify output stays upper triangular
- [ ] Test integration with Phase 1 output

---

## Phase 3: Sparse Component Finding
**Duration:** 2-3 days | **Depends on:** Phase 0

### 3.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_components.pyx`

- [ ] Rewrite `find_components()` to use CSR iteration
- [ ] Pre-compute CSC transpose for bidirectional access
- [ ] Replace O(n²) loops with CSR index iteration
- [ ] Maintain same API and output format
- [ ] Use Phase 0 transpose utilities

**Algorithm:**
```cython
A_csr = ensure_csr(A)
A_csc = ensure_csc(A)  # For incoming edges

for vertex in unvisited:
    # Outgoing edges
    for idx in range(A_csr.indptr[vertex], A_csr.indptr[vertex+1]):
        neighbor = A_csr.indices[idx]
        add_to_component(neighbor)
    
    # Incoming edges
    for idx in range(A_csc.indptr[vertex], A_csc.indptr[vertex+1]):
        neighbor = A_csc.indices[idx]
        add_to_component(neighbor)
```

**Test:** `tests/avos/test_components.py`
- [ ] Add sparse matrix tests (100K vertices, <1% density)
- [ ] Compare output with reference implementation
- [ ] Measure performance (expect O(V+E) scaling)
- [ ] Test edge cases: single vertices, disconnected components

---

## Phase 4: Sparse Canonical Permutation
**Duration:** 3-4 days | **Depends on:** Phase 0, Phase 3

### 4.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_ordering.pyx`

- [ ] Implement `_get_permutation_sparse(A, q, ids)`
- [ ] Use CSR iteration for ancestor counting
- [ ] Use CSC (transpose) for max relationship values
- [ ] Replace O(n²) loops with O(nnz) iteration
- [ ] Update `avos_canonical_ordering()` to use sparse version
- [ ] Add auto-detection: sparse vs dense

**Algorithm:**
```cython
A_csr = ensure_csr(A)
A_csc = ensure_csc(A)

# Ancestor counts from rows
for i in range(n):
    for idx in range(A_csr.indptr[i], A_csr.indptr[i+1]):
        value = A_csr.data[idx]
        ancestor_count[i] += MSB(value)

# Max relationships from columns  
for i in range(n):
    for idx in range(A_csc.indptr[i], A_csc.indptr[i+1]):
        value = A_csc.data[idx]
        max_rel[i] = max(max_rel[i], value)
```

**Test:** `tests/avos/test_ordering.py`
- [ ] Add sparse matrix tests
- [ ] Verify canonical property maintained
- [ ] Compare with reference implementation
- [ ] Measure performance (expect O(V+E+V log V))
- [ ] Integration test: Full pipeline from load to canonical

---

## Phase 5: Sparse AVOS Matrix Multiplication (GPU Prep)
**Duration:** 3-4 days | **Depends on:** Phase 0

### 5.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_avos_matmul.pyx`

- [ ] Create `_avos_matmul.pyx` file
- [ ] Implement `sparse_avos_matmul(A_csr, B_csr)`
- [ ] CSR × CSR multiplication with AVOS semiring
- [ ] Use avos_sum for aggregation
- [ ] Use avos_product for path composition
- [ ] Handle sparse structure efficiently

**Algorithm:**
```cython
# For each row i in A
for i in range(A.shape[0]):
    # For each k where A[i,k] != 0
    for k_idx in range(A.indptr[i], A.indptr[i+1]):
        k = A.indices[k_idx]
        a_ik = A.data[k_idx]
        
        # For each j where B[k,j] != 0
        for j_idx in range(B.indptr[k], B.indptr[k+1]):
            j = B.indices[j_idx]
            b_kj = B.data[j_idx]
            
            # C[i,j] = avos_sum(C[i,j], avos_product(a_ik, b_kj))
```

**Test:** `tests/sparse/test_avos_matmul.py`
- [ ] Create test file
- [ ] Test A ⊗ A = A²
- [ ] Compare with reference matmul
- [ ] Test repeated squaring for closure
- [ ] Benchmark against Floyd-Warshall

---

## Phase 6: Adaptive Closure Strategy
**Duration:** 1-2 days | **Depends on:** All previous phases

### 6.1: Implementation
**File:** `redblackgraph/sparse/csgraph/transitive_closure.py`

- [ ] Implement `transitive_closure_adaptive(A, method='auto')`
- [ ] Decision logic:
  ```python
  if multiple_components:
      use component_wise_closure()
  elif very_sparse and large:
      use dijkstra()
  elif upper_triangular:
      use floyd_warshall(assume_upper_triangular=True)
  else:
      use floyd_warshall() # accept densification
  ```
- [ ] Integrate density monitoring
- [ ] Add user-configurable thresholds

**Test:** `tests/sparse/test_adaptive_closure.py`
- [ ] Create test file
- [ ] Test strategy selection on various graph types
- [ ] Verify correctness across all paths
- [ ] Measure performance improvements

---

## Integration & Documentation
**Duration:** 2-3 days

- [ ] Update main `README.md` with sparse capabilities
- [ ] Create sparse workflow examples in `examples/sparse_workflow.py`
- [ ] Update `scripts/compute_canonical_forms.py` to use sparse by default
- [ ] Add --sparse-only flag to prevent densification
- [ ] Performance comparison notebook
- [ ] Update CHANGELOG.md with all new features

---

## Final Validation
**Duration:** 2-3 days

- [ ] Run full test suite on all Python versions (3.10, 3.11, 3.12)
- [ ] Large-scale test: 100K vertex graph end-to-end
- [ ] Memory profiling: Verify linear memory usage
- [ ] Performance profiling: Verify algorithmic complexity
- [ ] Code review and cleanup
- [ ] Documentation review

---

## Success Criteria (Must All Pass)

### Functional
- [ ] All tests pass (unit, integration, regression)
- [ ] No densification in permutation operations
- [ ] Topological sort produces upper triangular output
- [ ] Component finding runs in O(V+E)
- [ ] Canonical ordering runs in O(V+E+V log V)

### Performance
- [ ] Handle 100K vertex graphs with <1% density
- [ ] Memory usage O(V+E), not O(V²)
- [ ] Upper triangular FW is 1.8-2x faster
- [ ] Component-wise closure dramatically faster on multi-component graphs

### Quality
- [ ] Code coverage >85%
- [ ] All docstrings complete
- [ ] Type hints where appropriate
- [ ] No warnings from linters

---

## Dependencies Summary

```
Phase 0 (Infrastructure) ─────┬─> Phase 1 (Topological Sort)
                              │                │
                              │                └─> Phase 2b (Upper Tri FW)
                              │
                              ├─> Phase 3 (Sparse Components)
                              │                │
                              │                ├─> Phase 2a (Component-wise)
                              │                └─> Phase 4 (Canonical)
                              │
                              └─> Phase 5 (AVOS MatMul)

Phase 2a + Phase 2b + Phase 4 ─> Phase 6 (Adaptive Strategy)
```

**Critical Path:** 0 → 3 → 4 → 6 (must be sequential)

**Can Parallelize:**
- Phase 1 and Phase 3 (after Phase 0)
- Phase 2a and Phase 2b (after their prerequisites)
- Phase 5 (independent, GPU prep)
