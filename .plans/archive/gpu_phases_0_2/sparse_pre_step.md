# Sparse Matrix Optimization Implementation Plan

## Executive Summary

Before implementing GPU acceleration, optimize the CPU sparse matrix pipeline to handle large graphs. Current bottleneck: canonical form computation converts sparse matrices to dense, defeating the purpose of sparse representations.

**Goal:** Enable canonical form computation on graphs with 100K+ vertices using sparse operations throughout.

## Current State Analysis

### What Works ✓
- **Sparse transitive closure**: Both Floyd-Warshall and Dijkstra implementations
  - `transitive_closure_floyd_warshall()`
  - `transitive_closure_dijkstra()`
- **Sparse canonical ordering API**: `avos_canonical_ordering()` exists in `_ordering.pyx`
- **Sparse components**: `find_components()` exists in `_components.pyx`

### Critical Bottleneck ✗

All sparse operations have **hidden O(n²) loops** that iterate over all cells instead of just non-zero entries:

```python
# In _ordering.pyx line 29-34
for i in vertices:
    for j in vertices:  # O(n²) - defeats sparse!
        if Am[i][j]:
            ancester_count_for_vertex[i] += MSB(Am[i][j])
```

```python
# In _components.pyx line 58-60
for j in vertices:  # O(n²) - defeats sparse!
    if not ((Am[vertex][j] == 0 and Am[j][vertex] == 0) or ...):
        vertices_added_to_component.add(j)
```

**Result:** For large graphs, these "sparse" operations run in O(n²) time and require O(n²) memory.

## Mathematical Foundation

### Can Topological Sort Precede Transitive Closure?

**Question:** Can we permute to upper triangular form before computing transitive closure?

**Answer:** YES for topological ordering, NO for canonical ordering.

#### Topological Ordering (Pre-Closure) ✓
```
topological_sort(A) → P
(P·A·P⁻¹)* = P·A*·P⁻¹
```
- Requires only O(V+E) DFS traversal
- Works on original graph (no closure needed)
- Produces upper triangular form
- Enables faster closure computation

#### Canonical Ordering (Post-Closure Only) ✗
```
canonical_sort(A*) → P
```
- Requires transitive closure properties:
  - Component sizes (needs full connectivity)
  - Ancestor counts (needs all transitive paths)
  - Maximum relationship values (needs transitive paths)
- Cannot be computed from original graph alone

### Upper Triangular Optimization

For upper triangular matrix (after topological sort):
```python
# Standard Floyd-Warshall: O(n³)
for k in range(n):
    for i in range(n):
        for j in range(n):
            W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))

# Optimized for upper triangular: ~O(n³/2)
for k in range(n):
    for i in range(k+1):      # Only i ≤ k (skip lower triangle)
        for j in range(k, n):  # Only j ≥ k (skip lower triangle)
            W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))
```

**Benefit:** ~50% reduction in operations + better cache locality

## Implementation Phases

### Phase 1: Sparse Topological Sort (Foundation)
**Priority:** CRITICAL  
**Complexity:** LOW  
**Impact:** HIGH

#### Deliverables
1. **New file:** `redblackgraph/sparse/csgraph/_topological_sort.pyx`
   ```cython
   @cython.boundscheck(False)
   @cython.wraparound(False)
   def topological_sort(A) -> np.ndarray:
       """
       Topological sort using CSR sparse matrix format.
       
       Time: O(V + E) instead of O(V²)
       Space: O(V) for color/order arrays
       
       Parameters
       ----------
       A : CSR matrix or rb_array
           Input adjacency matrix (does not need to be transitively closed)
           
       Returns
       -------
       order : np.ndarray
           Permutation array representing topological ordering
       """
   ```

2. **Implementation details:**
   - DFS using CSR `indptr` and `indices` arrays
   - Iterative (stack-based) to avoid recursion limits
   - Handle both CSR matrices and rb_array types
   - Match reference implementation behavior for testing

3. **API addition:** `topological_ordering()` wrapper
   ```python
   def topological_ordering(A) -> Ordering:
       """
       Relabel graph to upper triangular form using topological sort.
       
       Does NOT require transitive closure.
       Result is upper triangular but not canonical.
       """
       ordering = topological_sort(A)
       return Ordering(permute(A, ordering), ordering, dict())
   ```

4. **Export:** Add to `redblackgraph/sparse/csgraph/__init__.py`

5. **Tests:** `tests/sparse/test_topological_sort.py`
   - Compare against reference implementation
   - Test on sparse CSR matrices
   - Test on rb_array
   - Verify upper triangular property
   - Test with multiple components

#### Why This First?
- No dependencies on other sparse optimizations
- Immediate benefit: enables pre-ordering
- Simplest to implement correctly
- Foundation for Phase 2 optimization
- Can be tested independently

---

### Phase 2: Upper Triangular Floyd-Warshall
**Priority:** HIGH  
**Complexity:** MEDIUM  
**Impact:** HIGH

#### Deliverables
1. **Modify:** `redblackgraph/sparse/csgraph/_shortest_path.pyx`
   - Add `assume_upper_triangular` parameter to floyd_warshall
   - Implement optimized triple loop (skip lower triangle)
   - Maintain correctness on diagonal and upper triangle

2. **API enhancement:**
   ```python
   def shortest_path(csgraph, method='auto', 
                     assume_upper_triangular=False, ...):
       """
       If assume_upper_triangular=True and method='FW':
           - Skip computations in lower triangle
           - ~50% reduction in operations
           - Result remains upper triangular
       """
   ```

3. **Update:** `transitive_closure_floyd_warshall()` wrapper
   ```python
   def transitive_closure_floyd_warshall(R, assume_upper_triangular=False):
       return transitive_closure(R, method="FW", 
                                assume_upper_triangular=assume_upper_triangular)
   ```

4. **Tests:** Add upper triangular tests to existing test suite
   - Verify correctness on pre-sorted matrices
   - Measure speedup (expect ~1.8-2x)
   - Ensure result stays upper triangular

#### Performance Target
- 2x speedup on transitive closure for large matrices
- Better cache performance due to spatial locality

---

### Phase 3: True Sparse Component Finding
**Priority:** HIGH  
**Complexity:** MEDIUM  
**Impact:** HIGH

#### Current Problem
```python
# _components.pyx line 58-60
for j in vertices:  # Checks ALL n vertices!
    if not ((Am[vertex][j] == 0 and Am[j][vertex] == 0) or ...):
```

#### Solution
```cython
@cython.boundscheck(False)
@cython.wraparound(False)
def find_components(A, q=None):
    """
    True sparse component identification.
    Time: O(V + E) instead of O(V²)
    """
    # Convert to CSR if needed
    cdef ITYPE_t[:] indptr = A.indptr
    cdef ITYPE_t[:] indices = A.indices
    
    for vertex in unvisited:
        # Only examine actual edges:
        for idx in range(indptr[vertex], indptr[vertex+1]):
            neighbor = indices[idx]
            # Add neighbor to component
        
        # Also check incoming edges (transpose)
        for idx in range(indptr_T[vertex], indptr_T[vertex+1]):
            neighbor = indices_T[idx]
            # Add neighbor to component
```

#### Deliverables
1. **Modify:** `redblackgraph/sparse/csgraph/_components.pyx`
   - Rewrite to use CSR iteration
   - Handle both directions (need transpose or COO format)
   - Maintain same API and output

2. **Handle edge cases:**
   - Single-vertex components
   - Disconnected graphs
   - Self-loops

3. **Tests:** Enhance `tests/avos/test_components.py`
   - Large sparse matrices (100K vertices, <1% density)
   - Compare output with reference implementation
   - Measure performance improvement

#### Performance Target
- O(V + E) time complexity
- Handle 100K vertex graphs with <1% density efficiently

---

### Phase 4: Sparse Canonical Permutation
**Priority:** MEDIUM  
**Complexity:** HIGH  
**Impact:** MEDIUM

#### Current Problem
```python
# _ordering.pyx _get_permutation line 29-34
for i in vertices:
    for j in vertices:  # O(n²)
        if Am[i][j]:
            ancester_count_for_vertex[i] += MSB(Am[i][j])
```

#### Solution
```cython
def _get_permutation_sparse(A, q, ids):
    """
    Compute permutation using sparse iteration.
    
    Only examine non-zero entries in CSR format.
    Time: O(V + E) for counting, O(V log V) for sorting
    """
    cdef ITYPE_t[:] indptr = A.indptr
    cdef ITYPE_t[:] indices = A.indices
    cdef DTYPE_t[:] data = A.data
    
    # Compute ancestor counts from non-zero entries only
    for i in range(n):
        for idx in range(indptr[i], indptr[i+1]):
            j = indices[idx]
            value = data[idx]
            if value != 0:
                ancester_count_for_vertex[i] += MSB(value)
    
    # Compute max relationships from transpose
    # (or maintain both CSR and CSC representations)
```

#### Deliverables
1. **Modify:** `redblackgraph/sparse/csgraph/_ordering.pyx`
   - Add `_get_permutation_sparse()` function
   - Use CSR iteration for counting
   - Maintain backward compatibility

2. **Decision:** Handle transpose efficiently
   - Option A: Compute CSC (transpose) once
   - Option B: Use COO format for bidirectional access
   - Option C: Two passes (CSR then CSR.T)

3. **Update:** `avos_canonical_ordering()`
   - Auto-detect sparse vs dense
   - Use appropriate implementation

4. **Tests:** `tests/avos/test_ordering.py`
   - Large sparse matrices
   - Verify canonical property maintained
   - Compare with reference output

#### Performance Target
- O(V + E + V log V) instead of O(V²)
- Handle 100K vertex graphs efficiently

---

## Workflow Comparison

### Current Workflow (Dense)
```python
A = load_graph()                    # Sparse
A_star = transitive_closure(A)      # Dense O(n³)
canonical = avos_canonical_ordering(A_star)  # Dense O(n²)
# FAILS: Out of memory for n > 10K
```

### After Phase 1 & 2 (Topological Pre-ordering)
```python
A = load_sparse_graph()                      # Sparse
ordering = topological_ordering(A)           # Sparse O(V+E)
A_triangular = ordering.A                    # Upper triangular
A_star = transitive_closure_floyd_warshall(
    A_triangular, 
    assume_upper_triangular=True             # 2x faster O(n³/2)
)
# Still converts to dense for closure
```

### After Phase 3 & 4 (Fully Sparse)
```python
A = load_sparse_graph()                      # Sparse
ordering = topological_ordering(A)           # Sparse O(V+E)
A_triangular = ordering.A                    # Sparse, upper triangular
A_star = transitive_closure_dijkstra(A_triangular)  # Sparse O(V²log V + VE)
canonical = avos_canonical_ordering(A_star)  # Sparse O(V+E+V log V)
# SUCCESS: Can handle 100K+ vertices
```

---

## Testing Strategy

### Unit Tests
Each phase gets dedicated test file:
- `tests/sparse/test_topological_sort.py`
- `tests/sparse/test_upper_triangular_closure.py`
- `tests/sparse/test_sparse_components.py`
- `tests/sparse/test_sparse_ordering.py`

### Integration Tests
- Compare sparse vs reference implementations
- Verify output equivalence (up to permutation)
- Test on real genealogy datasets

### Performance Tests
Create benchmark suite:
```python
# tests/performance/test_sparse_scaling.py
@pytest.mark.parametrize("n", [1000, 5000, 10000, 50000, 100000])
@pytest.mark.parametrize("density", [0.001, 0.01, 0.1])
def test_sparse_canonical_scaling(n, density):
    """Verify O(V+E) scaling for sparse operations"""
```

### Regression Tests
- Ensure canonical property maintained
- Verify graph isomorphism
- Check component identification correctness

---

## Success Metrics

### Phase 1 Success
- ✓ Topological sort works on 100K vertex sparse graphs
- ✓ O(V+E) time complexity verified
- ✓ Upper triangular property guaranteed
- ✓ All tests pass vs reference implementation

### Phase 2 Success
- ✓ Floyd-Warshall 1.8-2x faster on upper triangular
- ✓ Maintains upper triangular property through closure
- ✓ Correctness verified on test suite

### Phase 3 Success
- ✓ Component finding handles 100K vertices efficiently
- ✓ O(V+E) time complexity verified
- ✓ Output matches reference implementation

### Phase 4 Success
- ✓ End-to-end canonical form on 100K vertex graphs
- ✓ Memory usage linear in edges, not vertices²
- ✓ Canonical property verified
- ✓ Ready for GPU acceleration

---

## Critical Gap Analysis: True Sparse Requirements

The 4 phases above are necessary but **not sufficient** for true sparse implementations. Several critical infrastructure pieces are missing that would cause silent densification.

### Gap 1: Sparse Matrix Permutation ⚠️ BLOCKER

**Current Issue:**
```python
# redblackgraph/sparse/csgraph/_permutation.pyx line 14
cdef np.ndarray B = np.zeros(A.shape, dtype=DTYPE)  # CREATES DENSE OUTPUT!
```

The `permute()` function creates a **dense n×n matrix** regardless of input sparsity.

**Impact:** Every call to `topological_ordering()` or `avos_canonical_ordering()` **densifies the matrix**, defeating the entire purpose of sparse operations!

**Required Implementation:**
```cython
def permute_sparse(A_csr, p):
    """
    Sparse-to-sparse permutation preserving CSR format.
    
    Algorithm:
    1. Create new index arrays with remapped indices
    2. For each row i in output, source is row p[i] of input
    3. For each column j, destination is p[j]
    4. Build new CSR with permuted structure
    
    Time: O(nnz) where nnz = number of non-zero entries
    Space: O(nnz) - no dense intermediate
    """
    cdef int n = A_csr.shape[0]
    cdef ITYPE_t[:] indptr_in = A_csr.indptr
    cdef ITYPE_t[:] indices_in = A_csr.indices
    cdef DTYPE_t[:] data_in = A_csr.data
    
    # Allocate output arrays
    cdef np.ndarray[ITYPE_t] indptr_out = np.zeros(n+1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t] indices_out = np.zeros(A_csr.nnz, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t] data_out = np.zeros(A_csr.nnz, dtype=DTYPE)
    
    # Build permuted CSR structure
    # ... (detailed implementation)
    
    return rb_matrix((data_out, indices_out, indptr_out), shape=(n, n))
```

**Priority:** CRITICAL - Must implement before Phase 1

---

### Gap 2: Floyd-Warshall Dense Conversion ⚠️ BLOCKER

**Current Issue:**
```python
# _shortest_path.pyx line 299-301
dist_matrix = validate_graph(csgraph, directed, DTYPE,
                             csr_output=False,  # ← FORCES DENSE!
                             copy_if_dense=not overwrite)
```

Floyd-Warshall **always converts sparse to dense** via `validate_graph(..., csr_output=False)`.

**Reality Check:** 
- Floyd-Warshall on general graphs: Fills 50-90% of entries (densification acceptable)
- Floyd-Warshall on DAGs (especially upper triangular): Can stay relatively sparse
- Dijkstra's algorithm: Naturally preserves sparsity

**Required Decisions:**
1. **Accept densification for Floyd-Warshall on full graphs** (it's inherently dense)
2. **Use Dijkstra for truly sparse workflows** (already implemented)
3. **Component-wise processing** to limit densification scope

**Strategy:**
```python
def transitive_closure_adaptive(A, method='auto'):
    """
    Adaptive closure strategy based on graph structure.
    
    - Small graphs (<10K): Allow densification, use Floyd-Warshall
    - Large sparse graphs: Use Dijkstra (stays sparse)
    - Upper triangular: Use optimized Floyd-Warshall (controlled densification)
    - Multiple components: Process per-component
    """
    components = find_components_sparse(A)
    
    if len(components) > 1:
        # Process each component separately (smaller matrices)
        return component_wise_closure(A, components, method)
    elif A.nnz / (A.shape[0] ** 2) < 0.01 and A.shape[0] > 10000:
        # Very sparse large graph: Use Dijkstra
        return transitive_closure_dijkstra(A)
    else:
        # Accept densification for Floyd-Warshall
        return transitive_closure_floyd_warshall(A)
```

**Priority:** HIGH - Needed for Phase 2

---

### Gap 3: CSR Iteration Primitives ⚠️ FOUNDATION

**Missing:** Low-level Cython utilities for efficient CSR iteration.

**Required:**
```cython
# New file: redblackgraph/sparse/csgraph/_csr_utils.pxi

cdef inline void iter_row_edges(
    ITYPE_t[:] indptr,
    ITYPE_t[:] indices, 
    DTYPE_t[:] data,
    int vertex,
    void* callback_data,
    void (*callback)(int neighbor, DTYPE_t value, void* data)
):
    """
    Iterate over outgoing edges from vertex.
    Only examines non-zero entries.
    """
    cdef int idx, neighbor
    cdef DTYPE_t value
    
    for idx in range(indptr[vertex], indptr[vertex+1]):
        neighbor = indices[idx]
        value = data[idx]
        callback(neighbor, value, callback_data)

# Alternative: Generator/iterator pattern
cdef class CSRRowIterator:
    """Iterator for edges in a CSR row"""
    cdef ITYPE_t[:] indices
    cdef DTYPE_t[:] data
    cdef int start, end, pos
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.pos >= self.end:
            raise StopIteration
        neighbor = self.indices[self.pos]
        value = self.data[self.pos]
        self.pos += 1
        return neighbor, value
```

**Use Cases:**
- Topological sort (Phase 1)
- Component finding (Phase 3)
- Canonical permutation (Phase 4)
- All sparse graph algorithms

**Priority:** CRITICAL - Foundation for all phases

---

### Gap 4: Sparse Format Strategy

**Missing:** Clear strategy for when to use which format.

| Operation | Input Format | Output Format | Rationale |
|-----------|--------------|---------------|-----------|
| Topological sort | CSR | CSR | Forward edge iteration |
| Component finding | CSR + CSC | Vector | Need both directions |
| Permutation | CSR | CSR | Preserve sparsity |
| Floyd-Warshall (general) | CSR → Dense | Dense | Fills most entries |
| Floyd-Warshall (upper tri) | CSR | CSR/Dense | Controlled fill |
| Dijkstra | CSR | CSR | Stays sparse |
| Matrix output | CSR | CSR | Standard format |

**Required Utilities:**
```python
def ensure_csr(A):
    """Convert to CSR if not already"""
    if isinstance(A, rb_matrix):
        return A
    return rb_matrix(A)

def ensure_csc(A):
    """Convert to CSC for column access"""
    return A.tocsc() if hasattr(A, 'tocsc') else A.T.tocsr()

def to_dense_if_needed(A, threshold=0.5):
    """Convert to dense if density exceeds threshold"""
    density = A.nnz / (A.shape[0] * A.shape[1])
    if density > threshold:
        return A.toarray()
    return A
```

**Priority:** HIGH - Needed throughout

---

### Gap 5: Efficient Transpose Operations

**Problem:** Many operations need both row-wise and column-wise access:
- Component finding: Check both `A[i][j]` and `A[j][i]`
- Canonical permutation: Ancestors in columns, descendants in rows

**Current Approach:** Iterate over all n² cells (defeats sparsity!)

**Solution Options:**

**Option A: Pre-compute CSC transpose**
```python
def find_components_sparse(A):
    A_csr = ensure_csr(A)
    A_csc = A_csr.tocsc()  # One-time O(nnz) conversion
    
    # Now have efficient row AND column access
    for i in range(n):
        # Outgoing: A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]]
        # Incoming: A_csc.indices[A_csc.indptr[i]:A_csc.indptr[i+1]]
```

**Option B: COO format for bidirectional iteration**
```python
A_coo = A.tocoo()  # (row, col, data) triplets
# Can iterate and check both directions
```

**Option C: Build transpose indices on-the-fly**
```cython
cdef build_csr_transpose(A_csr):
    """Build CSR representation of A^T in O(nnz) time"""
    # Single pass through data to build transposed structure
```

**Recommendation:** Option A (pre-compute CSC) - simplest and most efficient for multiple accesses.

**Priority:** HIGH - Needed for Phases 3 & 4

---

### Gap 6: Component-Wise Processing

**Observation:** Large graphs often have:
- Many small disconnected components
- A few large connected components
- Sparse connections between components (zero by definition)

**Optimization Strategy:**
```python
def component_wise_closure(A, components=None):
    """
    Compute transitive closure per-component.
    
    Benefits:
    - Lower peak memory: O(max_component_size²) not O(n²)
    - Parallelizable across components
    - Sparse between components (exactly zero)
    
    Time: Σ O(k³) for each component of size k
          << O(n³) when components are small
    """
    if components is None:
        components = find_components_sparse(A)
    
    results = []
    for comp_id, comp_vertices in enumerate(components):
        # Extract submatrix for this component
        A_comp = extract_submatrix(A, comp_vertices)
        
        # Compute closure on small matrix (can densify if needed)
        A_comp_star = transitive_closure(A_comp)
        
        results.append((comp_vertices, A_comp_star))
    
    # Merge results - sparse between components
    return merge_component_closures(results, A.shape[0])
```

**Example:** 100K vertex graph with 1000 components of avg size 100:
- Full closure: O((100K)³) = 10¹⁵ operations
- Component-wise: 1000 × O(100³) = 10⁹ operations (1 million times faster!)

**Priority:** HIGH - Enables large-scale processing

---

### Gap 7: Density Monitoring

**Problem:** Operations silently densify without warning or control.

**Required:**
```python
class DensityMonitor:
    """Track and control sparsity throughout computation"""
    
    def __init__(self, warn_threshold=0.1, error_threshold=None):
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        self.history = []
    
    def check(self, A, operation_name):
        """Monitor density and warn/error as needed"""
        density = self.get_density(A)
        self.history.append((operation_name, density))
        
        if self.error_threshold and density > self.error_threshold:
            raise DensificationError(
                f"{operation_name} exceeded density limit: "
                f"{density:.2%} > {self.error_threshold:.2%}"
            )
        
        if density > self.warn_threshold:
            warnings.warn(
                f"{operation_name} density: {density:.2%}",
                DensificationWarning
            )
        
        return density
    
    @staticmethod
    def get_density(A):
        if hasattr(A, 'nnz'):
            return A.nnz / (A.shape[0] * A.shape[1])
        return 1.0  # Dense
```

**Usage:**
```python
monitor = DensityMonitor(warn_threshold=0.1, error_threshold=0.5)

A_topo = topological_ordering(A)
monitor.check(A_topo.A, "topological_ordering")  # Should stay sparse

A_star = transitive_closure(A_topo.A)
monitor.check(A_star, "transitive_closure")  # May densify (acceptable)

print(monitor.history)  # See density progression
```

**Priority:** MEDIUM - Quality of life, helps debugging

---

### Gap 8: Sparse AVOS Matrix Multiplication

**Current:** Matrix multiplication uses scipy's generic sparse matmul (doesn't understand AVOS).

**Required:** AVOS-specific sparse matrix multiplication:
```cython
def sparse_avos_matmul(A_csr, B_csr):
    """
    Compute C = A ⊗ B using AVOS semiring on sparse CSR matrices.
    
    Standard algorithm:
        C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
    
    AVOS specialization:
        C[i,j] = avos_sum_k (avos_product(A[i,k], B[k,j]))
    
    Sparse optimization:
        Only compute for k where both A[i,k] ≠ 0 AND B[k,j] ≠ 0
    
    Time: O(nnz(A) × avg_row_nnz(B))
    """
    # Sparse CSR × CSR multiplication with AVOS operations
    # Similar to scipy.sparse but with avos_sum/avos_product
```

**Use Cases:**
- Matrix powers: A² = A ⊗ A, A⁴ = A² ⊗ A²
- Repeated squaring for closure (alternative to Floyd-Warshall)
- GPU preparation (natural parallel algorithm)

**Priority:** MEDIUM - Alternative algorithm, critical for GPU

---

### Gap 9: Upper Triangular Sparse Storage

**Opportunity:** Upper triangular matrices can use specialized storage.

**Benefits:**
- 50% memory reduction (only store upper triangle)
- Faster iteration (skip half the matrix explicitly)
- Natural for DAGs after topological sort

**Implementation:**
```python
class UpperTriangularCSR(rb_matrix):
    """
    Specialized CSR for upper triangular matrices.
    Assumes A[i,j] = 0 for all i > j
    Optimized storage and operations.
    """
    
    def __init__(self, data, indices, indptr):
        # Verify upper triangular property
        super().__init__((data, indices, indptr))
        self._verify_upper_triangular()
    
    def _verify_upper_triangular(self):
        """Assert no entries below diagonal"""
        for i in range(self.shape[0]):
            for idx in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[idx]
                assert j >= i, f"Entry at ({i},{j}) violates upper triangular"
```

**Priority:** LOW - Optimization, not required for correctness

---

## Phase 0: Sparse Infrastructure (NEW - FOUNDATION)

Must be implemented **before** Phases 1-4.

### Deliverables

1. **CSR Iteration Primitives** (`_csr_utils.pxi`)
   - Row edge iterator
   - Column edge iterator (via CSC)
   - Efficient sparse loops

2. **Sparse Permutation** (`_permutation.pyx` enhancement)
   - `permute_sparse()` function
   - CSR → CSR with index remapping
   - Preserve sparsity through reordering

3. **Format Conversion Utilities** (`_sparse_format.py`)
   - `ensure_csr()`, `ensure_csc()`
   - `to_dense_if_needed(threshold)`
   - Format detection and conversion

4. **Transpose Utilities** (`_csr_utils.pxi`)
   - `build_csr_transpose()`
   - Efficient CSR → CSC conversion
   - Cached transpose for repeated access

5. **Density Monitoring** (`_density.py`)
   - `DensityMonitor` class
   - Warning/error thresholds
   - History tracking

6. **Component Extraction** (`_components.pyx` enhancement)
   - `extract_submatrix(A, vertices)` - sparse extraction
   - `merge_component_matrices(components)` - sparse merge
   - Preserve sparsity in extraction/merge

### Tests
- `tests/sparse/test_csr_utils.py`
- `tests/sparse/test_sparse_permutation.py`
- `tests/sparse/test_format_conversion.py`
- `tests/sparse/test_density_monitoring.py`

### Success Criteria
- ✓ Permutation preserves sparsity (no densification)
- ✓ CSR iteration O(nnz) verified
- ✓ Format conversions efficient and correct
- ✓ All utilities tested independently

### Timeline
**Estimated:** 4-6 days

This is the **foundation** for all subsequent sparse work.

---

## Revised Implementation Order

### Phase 0: Sparse Infrastructure (4-6 days) - **MUST DO FIRST**
All foundation utilities described above.

### Phase 1: Sparse Topological Sort (1-2 days)
Now uses Phase 0 primitives:
- CSR iteration for edge traversal
- Sparse permutation for output
- Density monitoring

### Phase 2a: Component-Wise Processing (2-3 days)
- Uses Phase 0 component extraction
- Per-component closure
- Sparse merge

### Phase 2b: Upper Triangular Floyd-Warshall (2-3 days)
- Optimized loop structure
- Controlled densification
- Works on output of Phase 1

### Phase 3: Sparse Component Finding (2-3 days)
- Uses Phase 0 CSR iteration
- Uses transpose utilities
- Truly O(V + E)

### Phase 4: Sparse Canonical Permutation (3-4 days)
- Uses Phase 0 CSR iteration
- Uses sparse permutation
- Uses transpose for column access

### Phase 5: Sparse AVOS MatMul (3-4 days) - **GPU Prep**
- CSR × CSR with AVOS semiring
- Alternative closure algorithm
- Foundation for GPU

**Revised Total:** 17-25 days (including Phase 0)

---

## When Densification is Acceptable

**Accept densification for:**
- ✓ Small graphs (< 10K vertices)
- ✓ Within single components during closure
- ✓ Floyd-Warshall results (inherently dense for most graphs)
- ✓ Final output if needed for GPU transfer
- ✓ When density already > 50%

**Never densify for:**
- ✗ Initial graph representation
- ✗ Permutation operations (now preserved via Phase 0)
- ✗ Component identification
- ✗ Between-component storage
- ✗ Intermediate topological ordering

**Monitor and warn for:**
- ⚠ Transitive closure on large sparse graphs
- ⚠ Operations that cross density thresholds
- ⚠ Unexpected density increases

The key is **controlled densification** with **monitoring and user choice**.

---

## Critical Path Summary

```
Phase 0 (Infrastructure)
    ├─> Enables Phase 1 (Topological Sort)
    ├─> Enables Phase 3 (Sparse Components)
    └─> Enables Phase 4 (Sparse Canonical)

Phase 1 (Topological Sort)
    └─> Enables Phase 2b (Upper Tri FW)

Phase 3 (Sparse Components)
    ├─> Enables Phase 2a (Component-wise)
    └─> Enables Phase 4 (Sparse Canonical)

Phase 2a + Phase 2b (Closure Optimizations)
    └─> Together enable large-scale closure

Phase 4 (Sparse Canonical)
    └─> Complete sparse pipeline

Phase 5 (Sparse MatMul)
    └─> GPU readiness
```

**Cannot skip Phase 0!** It's the foundation everything else depends on.

---

## GPU Readiness

These optimizations prepare for GPU implementation:

1. **Sparse CSR format** → Natural GPU representation
2. **Upper triangular** → Reduces GPU memory pressure
3. **Component-wise operations** → Enables GPU parallelization per component
4. **Reduced memory footprint** → More fits in GPU memory

After this work, GPU implementation can focus on:
- Parallel sparse matrix operations
- GPU-accelerated Floyd-Warshall on triangular matrices
- Parallel topological sort
- Batch processing of components

---

## Risk Mitigation

### Risk: Transpose Performance
**Mitigation:** Benchmark CSR.T vs maintaining CSC vs COO format

### Risk: Dense Conversion Still Needed
**Mitigation:** Identify which operations truly need dense; consider block-sparse

### Risk: Complexity Creep
**Mitigation:** Implement phases sequentially; each must pass tests before next

### Risk: API Breaking Changes
**Mitigation:** Maintain backward compatibility; add parameters, don't remove

---

## Dependencies

### Phase 1: None (independent)
### Phase 2: Requires Phase 1 for testing, but can be implemented in parallel
### Phase 3: None (independent)
### Phase 4: Requires Phase 3 (needs sparse components)

**Parallelization opportunity:** Phases 1 & 3 can be developed simultaneously.

---

## Timeline Estimate

- **Phase 1:** 1-2 days (straightforward port from reference)
- **Phase 2:** 2-3 days (modify existing code, careful testing)
- **Phase 3:** 3-4 days (bidirectional iteration tricky)
- **Phase 4:** 4-5 days (most complex, needs careful sorting logic)

**Total:** 10-14 days for complete sparse pipeline

---

## Document Analysis Summary

### Completeness: GOOD ✓
- All critical infrastructure gaps identified
- Mathematical foundation well-explained
- Clear success metrics defined

### Issues Found:
1. **Timeline inconsistency:** States both "10-14 days" and "17-25 days"
2. **Phase numbering:** Inconsistent (0, 1, 2a, 2b, 3, 4, 5)
3. **Redundancy:** Dependencies mentioned in 3 different places
4. **Missing:** Build system updates (meson.build)
5. **Missing:** Specific algorithm details for component extraction/merge

### Recommendation:
See SPARSE_IMPLEMENTATION_CHECKLIST.md for execution-ordered version with checkboxes.
3. **Measure:** Benchmark to confirm O(V+E) scaling
4. **Iterate:** Proceed to Phase 2 if Phase 1 successful

---

## References

- Current implementations:
  - `redblackgraph/reference/topological_sort.py` (reference)
  - `redblackgraph/reference/ordering.py` (canonical ordering)
  - `redblackgraph/sparse/csgraph/_ordering.pyx` (sparse canonical - needs optimization)
  - `redblackgraph/sparse/csgraph/_components.pyx` (sparse components - needs optimization)
  
- Related tests:
  - `tests/reference/test_triangularization.py`
  - `tests/avos/test_ordering.py`
  - `tests/avos/test_components.py`
