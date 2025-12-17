# Sparse Matrix Implementation - Execution Checklist

**Goal:** Enable canonical form computation on 100K+ vertex graphs using truly sparse operations.

**Timeline:** 17-25 days total

**Existing test structure:**
- `tests/sparse/` - exists, contains `test_sparse_matmul.py`
- `tests/avos/` - existing component/ordering tests (must remain passing)

---

## Phase 0: Sparse Infrastructure (CRITICAL FOUNDATION) ✅ COMPLETE
**Duration:** 4-6 days | **Must complete before all other phases**

**Status:** Completed on 2024-11-24. All 202 tests passing.

> **Note:** Current implementations (`_components.pyx`, `_ordering.pyx`, `_permutation.pyx`) use dense
> `DTYPE_t[:, :]` typed memoryviews, forcing O(n²) iteration. The sparse versions require fundamentally
> different CSR/CSC index-based access patterns.

### 0.1: CSR Iteration Primitives
**File:** `redblackgraph/sparse/csgraph/_csr_utils.pxi` (new include file)

- [x] Create `_csr_utils.pxi` include file (`.pxi` files are Cython includes, not compiled modules)
- [x] Implement inline row edge iterator macro
  ```cython
  cdef inline iterate_row_edges(ITYPE_t[:] indptr, ITYPE_t[:] indices, DTYPE_t[:] data, int vertex)
  ```
- [x] Implement CSRRowIterator class for Python-accessible iteration
- [x] Add column edge iterator (requires CSC format)
- [x] Document usage patterns

**Test:** `tests/sparse/test_csr_utils.py`
- [x] Create test file
- [x] Test row iteration matches manual index access
- [x] Test iteration only visits non-zero entries
- [x] Benchmark: Verify O(nnz) not O(n²)

---

### 0.2: Sparse Matrix Permutation
**File:** `redblackgraph/sparse/csgraph/_permutation.pyx` (modify existing)

> **Current state:** Existing `permute()` uses dense memoryview `DTYPE_t[:, :]` at line 24-26,
> always allocates dense output at line 14.

- [x] Implement `permute_sparse(A_csr, p)` function
  - Algorithm:
    1. Create inverse permutation p_inv
    2. Count edges per output row
    3. Build output indptr from counts
    4. Iterate input rows in permuted order
    5. Remap column indices via p_inv
    6. Copy to output arrays
- [x] Handle upper triangular optimization flag
- [x] Update existing `permute()` to dispatch to sparse version
- [x] Preserve CSR format (no densification!)

**Test:** `tests/sparse/test_sparse_permutation.py`
- [x] Create test file
- [x] Test permutation preserves graph structure
- [x] Test NO densification occurs (check output.nnz == input.nnz)
- [x] Compare with dense permutation results
- [x] Test upper triangular case

---

### 0.3: Format Conversion Utilities
**File:** `redblackgraph/sparse/csgraph/_sparse_format.py` (new)

> **Note:** Leverage scipy.sparse utilities where possible (`isspmatrix_csr`, `csr_matrix.tocsc()`, etc.)
> rather than reimplementing. Focus on RBG-specific wrappers and density monitoring.

- [x] Create `_sparse_format.py` file
- [x] Implement `ensure_csr(A)` - thin wrapper around scipy, handle rb_matrix/rb_array types
- [x] Implement `ensure_csc(A)` - convert to CSC for column access
- [x] Implement `to_dense_if_needed(A, threshold)` - conditional densification with logging
- [x] Implement `get_density(A)` helper
- [x] Add format detection utilities for rb_matrix vs scipy.sparse types

**Test:** `tests/sparse/test_csr_utils.py` (consolidated)
- [x] Create test file
- [x] Test CSR/CSC round-trip
- [x] Test density threshold logic
- [x] Test format detection

---

### 0.4: Transpose Utilities
**File:** `redblackgraph/sparse/csgraph/_csr_utils.pxi`

- [x] Implement `build_csr_transpose(A_csr)` in Cython
  - O(nnz) single-pass algorithm
  - Build CSR format of A^T
- [x] Implement caching mechanism for repeated transpose (via scipy)
- [x] Add CSR → CSC conversion helper

**Test:** `tests/sparse/test_csr_utils.py` (consolidated)
- [x] Test transpose correctness
- [x] Test via scipy's efficient transpose

---

### 0.5: Density Monitoring
**File:** `redblackgraph/sparse/csgraph/_density.py`

- [x] Create `_density.py` file
- [x] Implement `DensityMonitor` class
  - `__init__(warn_threshold, error_threshold)`
  - `check(A, operation_name)` - monitor and warn/error
  - `get_density(A)` - compute current density
  - `history` - track density through pipeline
- [x] Implement `DensificationError` exception
- [x] Implement `DensificationWarning` warning class

**Test:** `tests/sparse/test_csr_utils.py` (consolidated)
- [x] Test warning triggers
- [x] Test error triggers
- [x] Test history tracking

---

### 0.6: Component Extraction & Merge
**File:** `redblackgraph/sparse/csgraph/_components.pyx`

- [x] Implement `extract_submatrix(A_csr, vertices)` - sparse extraction
  - Build mapping: old_id → new_id
  - Extract rows for specified vertices
  - Remap column indices
  - Return sparse submatrix
- [x] Implement `merge_component_matrices(components, n_total)`
  - Pre-allocate full matrix structure
  - Place each component at correct offset
  - Return full sparse matrix
- [x] Implement `find_components_sparse(A)` - O(V+E) component finding
- [x] Implement `get_component_vertices(labels)` - utility function

**Test:** `tests/sparse/test_component_extraction.py`
- [x] Create test file
- [x] Test extraction preserves subgraph
- [x] Test merge reconstructs full graph
- [x] Test round-trip: extract → merge → compare

---

### 0.7: Build System Updates
**Files:** `redblackgraph/sparse/csgraph/meson.build`, `__init__.py`

- [x] Add `_csr_utils.pxi` to meson.build includes (auto-included via .pyx)
- [x] Add `_sparse_format.py` to Python sources
- [x] Add `_density.py` to Python sources
- [x] Update `__init__.py` to export new utilities
- [x] Test build on clean environment

---

## Phase 1: Sparse Topological Sort ✅ COMPLETE
**Duration:** 1-2 days | **Depends on:** Phase 0

**Status:** Completed on 2024-11-25. All 25 tests passing.

### 1.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_topological_sort.pyx`

- [x] Create `_topological_sort.pyx` file
- [x] Implement `topological_sort(A)` function
  - Use Phase 0 CSR iteration primitives
  - Iterative DFS (stack-based, no recursion)
  - Handle both CSR and rb_array inputs
  - Return permutation array
- [x] Implement `topological_ordering(A)` wrapper
  - Call topological_sort
  - Use Phase 0 sparse permutation
  - Return Ordering object
- [x] Implement `is_upper_triangular(A)` helper function
- [x] Add to `__init__.py` exports

**Test:** `tests/sparse/test_topological_sort.py`
- [x] Create test file
- [x] Compare output with reference implementation
- [x] Test on sparse CSR matrices (100K vertices)
- [x] Verify upper triangular property of result
- [x] Test with multiple disconnected components
- [x] Benchmark: Confirm O(V+E) scaling

---

### 1.2: Build Integration
- [x] Add to `meson.build`
- [x] Update package `__init__.py`
- [x] Rebuild and test

---

## Phase 2a: Component-Wise Processing ✅ COMPLETE
**Duration:** 2-3 days | **Depends on:** Phase 0, Phase 3

**Status:** Completed on 2024-11-25. All 13 tests passing.

### 2a.1: Implementation
**File:** `redblackgraph/sparse/csgraph/transitive_closure.py`

- [x] Implement `component_wise_closure(A, components, method)`
  - Use Phase 0 extraction to get submatrices
  - Compute closure per component (can densify small matrices)
  - Use Phase 0 merge to reconstruct full matrix
  - Return sparse result (sparse between components)

**Test:** `tests/sparse/test_component_closure.py`
- [x] Create test file
- [x] Test on multi-component graph
- [x] Verify result equals full closure
- [x] Measure memory savings vs full graph
- [x] Test parallelization potential (mark for future)

---

## Phase 2b: Upper Triangular Floyd-Warshall ✅ COMPLETE
**Duration:** 2-3 days | **Depends on:** Phase 1

**Status:** Completed on 2024-11-25. All 13 tests passing.

### 2b.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_shortest_path.pyx`

- [x] Modify `_floyd_warshall_avos()` to accept `assume_upper_triangular` flag
- [x] Implement optimized triple loop:
  ```cython
  for k in range(N):
      for i in range(k+1):      # Only i ≤ k
          for j in range(k, N):  # Only j ≥ k
  ```
- [x] Update `floyd_warshall()` wrapper to accept parameter
- [x] Update `shortest_path()` to pass through parameter
- [x] Update `transitive_closure_floyd_warshall()` wrapper
- [x] Export `floyd_warshall` from `__init__.py`

**Test:** `tests/sparse/test_upper_triangular_closure.py`
- [x] Create test file
- [x] Test correctness on upper triangular matrices
- [x] Measure speedup (expect 1.8-2x)
- [x] Verify output stays upper triangular
- [x] Test integration with Phase 1 output

---

## Phase 3: Sparse Component Finding ✅ COMPLETE
**Duration:** 2-3 days | **Depends on:** Phase 0

**Status:** Completed on 2024-11-25. 14 new tests + 7 regression tests passing.

### 3.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_components.pyx` (modify existing)

> **Current O(n²) bottleneck** at lines 58-60:
> ```cython
> for j in vertices:
>     if not ((Am[vertex][j] == 0 and Am[j][vertex] == 0) or ...)
> ```
> This iterates all vertices instead of only non-zero edges.

- [x] Add `find_components_sparse(A_csr)` function (new function, preserve original for dense)
- [x] Pre-compute CSC transpose for bidirectional access
- [x] Replace O(n²) loop with CSR/CSC index iteration
- [x] Update `find_components()` to dispatch: sparse input → `find_components_sparse()`
- [x] Maintain same API and output format
- [x] Use Phase 0 transpose utilities

**Algorithm:**
```cython
A_csr = ensure_csr(A)
A_csc = ensure_csc(A)  # For incoming edges

for vertex in unvisited:
    # Outgoing edges - O(out_degree) not O(n)
    for idx in range(A_csr.indptr[vertex], A_csr.indptr[vertex+1]):
        neighbor = A_csr.indices[idx]
        add_to_component(neighbor)
    
    # Incoming edges - O(in_degree) not O(n)
    for idx in range(A_csc.indptr[vertex], A_csc.indptr[vertex+1]):
        neighbor = A_csc.indices[idx]
        add_to_component(neighbor)
```

**Test:** `tests/sparse/test_components.py` (new file)
- [x] Add sparse matrix tests (100K vertices, <1% density)
- [x] Compare output with dense `find_components()` reference
- [x] Measure performance (expect O(V+E) scaling)
- [x] Test edge cases: single vertices, disconnected components
- [x] Ensure existing `tests/avos/test_components.py` still passes (regression)

---

## Phase 4: Sparse Canonical Permutation ✅ COMPLETE
**Duration:** 3-4 days | **Depends on:** Phase 0, Phase 3

**Status:** Completed on 2024-11-25. All tests passing.

### 4.1: Implementation
**File:** `redblackgraph/sparse/csgraph/_ordering.pyx` (modify existing)

> **Current O(n²) bottleneck** at lines 29-34:
> ```cython
> for i in vertices:
>     for j in vertices:  # O(n²) iteration!
>         if Am[i][j]:
>             ancester_count_for_vertex[i] += MSB(Am[i][j])
> ```

- [x] Implement `_get_permutation_sparse(A_csr, q, ids)` (new function)
- [x] Use CSR iteration for ancestor counting (row traversal)
- [x] Use CSC (transpose) for max relationship values (column traversal)
- [x] Update `avos_canonical_ordering()` to dispatch based on input type
- [x] Preserve `_get_permutation()` for dense inputs (backwards compatibility)

**Algorithm:**
```cython
A_csr = ensure_csr(A)
A_csc = ensure_csc(A)

# Ancestor counts from rows - O(nnz) not O(n²)
for i in range(n):
    for idx in range(A_csr.indptr[i], A_csr.indptr[i+1]):
        value = A_csr.data[idx]
        ancestor_count[i] += MSB(value)

# Max relationships from columns - O(nnz) not O(n²)
for i in range(n):
    for idx in range(A_csc.indptr[i], A_csc.indptr[i+1]):
        value = A_csc.data[idx]
        max_rel[i] = max(max_rel[i], value)
```

**Test:** `tests/sparse/test_ordering.py` (new file)
- [x] Add sparse matrix tests
- [x] Verify canonical property maintained
- [x] Compare with dense `_get_permutation()` reference
- [x] Measure performance (expect O(V+E+V log V))
- [x] Integration test: Full pipeline from load to canonical
- [x] Ensure existing `tests/avos/test_ordering.py` still passes (regression)

---

## Phase 5: Sparse AVOS Matrix Multiplication ✅ N/A (Existing C++ Implementation)
**Duration:** N/A | **Depends on:** Phase 0

**Status:** Marked N/A on 2024-11-25. Existing C++ implementation in `rbm_math.h` already provides optimized AVOS sparse matmul via `rb_matrix` class operators (`@`, `*`).

### 5.1: Existing Implementation
**Files:** 
- `redblackgraph/sparse/sparsetools/rbm_math.h` - C++ AVOS operations
- `redblackgraph/sparse/rbm.py` - `rb_matrix` class with `__matmul__`

**Already Available:**
- [x] `rbm_matmat_pass1` and `rbm_matmat_pass2` in C++
- [x] `rb_matrix @ rb_matrix` for sparse AVOS multiplication
- [x] `avos_sum` and `avos_product` operations
- [x] Existing tests in `tests/sparse/test_sparse_matmul.py`

**Rationale:** Creating a pure Cython duplicate would be:
- Redundant (same functionality)
- Slower (C++ templates are highly optimized)
- More maintenance (two implementations to keep in sync)

For GPU implementation, CUDA/cuSPARSE would be used directly.

### 5.2: Bonus - Closure via Repeated Squaring
**File:** `redblackgraph/sparse/csgraph/transitive_closure.py`

- [x] Implement `transitive_closure_squaring(A)` using existing matmul
- [x] Compute A + A² + A⁴ + ... until convergence
- [x] O(log d) matrix multiplications where d = diameter

**Test:** Included in `tests/sparse/test_adaptive_closure.py`

---

## Phase 6: Adaptive Closure Strategy ✅ COMPLETE
**Duration:** 1-2 days | **Depends on:** All previous phases

**Status:** Completed on 2024-11-25. All tests passing.

### 6.1: Implementation
**File:** `redblackgraph/sparse/csgraph/transitive_closure.py`

- [x] Implement `transitive_closure_adaptive(A, method='auto')`
- [x] Decision logic:
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
- [x] Integrate density monitoring
- [x] Add user-configurable thresholds (density_threshold, component_threshold, size_threshold)
- [x] Implement `transitive_closure_squaring(A)` for repeated squaring closure

**Test:** `tests/sparse/test_adaptive_closure.py`
- [x] Create test file
- [x] Test strategy selection on various graph types
- [x] Verify correctness across all paths
- [x] Test all methods produce equivalent results

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
                              │                ├─> Phase 2a (Component-wise) ←── also needs closure algorithm
                              │                └─> Phase 4 (Canonical)
                              │
                              └─> Phase 5 (AVOS MatMul) ←── GPU prep, independent

Phase 2a + Phase 2b + Phase 4 ─> Phase 6 (Adaptive Strategy)
```

**Critical Path:** 0 → 3 → 4 → 6 (must be sequential)

**Can Parallelize:**
- Phase 1 and Phase 3 (after Phase 0)
- Phase 2a and Phase 2b (after their prerequisites)
- Phase 5 (independent, GPU prep)
