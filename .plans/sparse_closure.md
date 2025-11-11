# Sparse Transitive Closure Implementation

## Problem Statement

The current transitive closure implementation in `redblackgraph/sparse/csgraph/_shortest_path.pyx` creates dense output matrices, causing memory allocation failures for large sparse graphs.

### Current Behavior

**Location:** `redblackgraph/sparse/csgraph/_shortest_path.pyx:572`

```python
dist_matrix = np.zeros((len(indices), N), dtype=DTYPE).view(array)
```

This line allocates a dense `(N, N)` matrix to store shortest path distances, even when:
- The input graph is sparse (e.g., 10K edges in a 4.5M vertex graph)
- The output transitive closure will also be sparse (for DAGs, closure density ≈ input density × graph depth)
- The user explicitly chose sparse representation via `sparse_threshold`

### Memory Impact

For a graph with N=4,502,136 vertices:
- Dense matrix size: N × N × 4 bytes (int32) = **81 TB**
- Sparse matrix size (10K edges): ~10K × (8+8+4) bytes = **200 KB**
- **Memory ratio: 405,000,000:1**

This makes it impossible to compute transitive closure for large real-world graphs, even though the underlying computation could work with sparse data structures.

### Current Workaround

In `redblackgraph/util/graph_builder.py:46-53`, we skip canonical ordering for large graphs:

```python
if self.nv > self.sparse_threshold:
    logging.warning(f"Graph too large ({self.nv:,} vertices) for canonical ordering.")
    return list(range(self.nv))  # Identity ordering
```

This works but loses the benefits of canonical ordering for large graphs.

## Requirements

### Functional Requirements

1. **Sparse Input/Output**: Accept sparse CSR/CSC matrices, return sparse matrix
2. **Memory Efficiency**: Memory usage proportional to nnz (number of non-zeros), not N²
3. **Correctness**: Compute exact transitive closure (all reachable pairs)
4. **AVOS Algebra**: Support AVOS semiring operations, not just boolean reachability
5. **API Compatibility**: Drop-in replacement for current `transitive_closure()` function

### Performance Requirements

1. **Large Sparse Graphs**: Handle N > 1M vertices with nnz < 10M
2. **Time Complexity**: O(nnz × average_path_length) for typical genealogy DAGs
3. **Space Complexity**: O(nnz_closure) where nnz_closure is output non-zeros
4. **Scalability**: Should work on graphs where N² exceeds RAM but nnz_closure fits

### Quality Requirements

1. **No Regression**: Existing tests must pass
2. **Performance**: Should be faster than dense algorithm for sparse graphs
3. **Memory Safety**: No silent failures or swapping; fail fast with clear error messages
4. **Maintainability**: Code should be readable and well-documented

## Solution Approach

### Algorithm Selection

For DAGs (directed acyclic graphs), which are common in genealogy applications:

**Option 1: Topological Sort + Forward Pass**
- Sort vertices topologically
- For each vertex in topological order:
  - Union its closure with closures of all predecessors
- Complexity: O(N + nnz_closure)
- Best for: DAGs with low closure density

**Option 2: Warshall's Algorithm (Sparse)**
- Iterate k = 1 to N:
  - For each edge (i, k) in column k:
    - For each edge (k, j) in row k:
      - Add edge (i, j) with AVOS product
- Complexity: O(N × nnz_closure)
- Best for: Dense closures or general directed graphs

**Option 3: Iterative BFS/DFS**
- For each vertex v:
  - Perform BFS/DFS to find all reachable vertices
  - Record edges in closure
- Complexity: O(N × (nnz + nnz_closure))
- Best for: Very sparse closures

**Recommendation:** Start with Option 1 (topological sort) for DAGs since:
- Genealogy graphs are typically DAGs
- Single pass algorithm is simple and efficient
- Can fall back to Option 2 for graphs with cycles

### Data Structures

**Sparse Matrix Format:**
- Use CSR (Compressed Sparse Row) for efficient row access
- Use CSC (Compressed Sparse Column) for efficient column access
- Keep both representations if needed, or convert as needed

**Intermediate Storage:**
- Build output in COO (Coordinate) format during computation
- Convert to CSR/CSC at end for efficient return

**Memory Management:**
- Pre-allocate based on estimated closure size
- Use dynamic resizing if needed (e.g., Python lists or std::vector)
- Consider memory-mapped files for very large closures

## Detailed Design

### Phase 1: Sparse Transitive Closure for DAGs

#### API

```python
def sparse_transitive_closure(
    graph: csr_matrix,
    method: str = 'auto',
    overwrite: bool = False,
    dense_threshold: int = 1000
) -> csr_matrix:
    """
    Compute transitive closure maintaining sparsity.
    
    Parameters
    ----------
    graph : csr_matrix or csc_matrix
        Input adjacency matrix with AVOS semiring values
    method : str, optional
        Algorithm: 'auto', 'topological', 'warshall', 'iterative'
        Default 'auto' selects based on graph properties
    overwrite : bool, optional
        If True, may modify input matrix for efficiency
    dense_threshold : int, optional
        If N < threshold, fall back to dense algorithm
        
    Returns
    -------
    closure : csr_matrix
        Transitive closure in sparse format
        
    Raises
    ------
    CycleError
        If method='topological' and graph has cycles
    MemoryError
        If estimated closure size exceeds available memory
    """
```

#### Implementation: Topological Sort Approach

```python
def _sparse_closure_topological(graph: csr_matrix) -> csr_matrix:
    """
    Compute transitive closure using topological sort.
    Works only for DAGs.
    """
    N = graph.shape[0]
    
    # Step 1: Topological sort
    try:
        topo_order = topological_sort(graph)
    except CycleError:
        raise CycleError("Topological sort requires DAG")
    
    # Step 2: Initialize closure with direct edges
    # Use COO format for building
    row_list = []
    col_list = []
    data_list = []
    
    # Add all existing edges
    cx = graph.tocoo()
    row_list.extend(cx.row)
    col_list.extend(cx.col)
    data_list.extend(cx.data)
    
    # Step 3: Forward pass in topological order
    # For each vertex, inherit edges from predecessors
    closure_csr = csr_matrix((data_list, (row_list, col_list)), shape=(N, N))
    
    for v in topo_order:
        # Get all predecessors of v (vertices with edges to v)
        predecessors = closure_csr.getcol(v).nonzero()[0]
        
        # Get all successors of v (vertices reachable from v)
        successors_row = closure_csr.getrow(v)
        succ_cols = successors_row.nonzero()[1]
        succ_data = successors_row.data
        
        # For each predecessor u, add edges to all successors of v
        for u in predecessors:
            # Get AVOS value for u->v edge
            u_to_v = closure_csr[u, v]
            
            # Add u->w edges for all successors w of v
            for w_idx, w in enumerate(succ_cols):
                v_to_w = succ_data[w_idx]
                # AVOS product: u->w = (u->v) ⊗ (v->w)
                new_val = avos_product(u_to_v, v_to_w)
                
                # Update or insert edge
                existing = closure_csr[u, w]
                if existing == 0:
                    row_list.append(u)
                    col_list.append(w)
                    data_list.append(new_val)
                else:
                    # AVOS sum: take max or appropriate operation
                    combined = avos_sum(existing, new_val)
                    # Update in place (requires rebuilding matrix periodically)
        
        # Rebuild CSR periodically to maintain efficiency
        if len(row_list) > N * 10:  # Heuristic threshold
            closure_csr = csr_matrix((data_list, (row_list, col_list)), shape=(N, N))
            closure_csr.sum_duplicates()
            row_list.clear()
            col_list.clear()
            data_list.clear()
            # Re-extract for next iteration
            cx = closure_csr.tocoo()
            row_list.extend(cx.row)
            col_list.extend(cx.col)
            data_list.extend(cx.data)
    
    # Final conversion
    closure_csr = csr_matrix((data_list, (row_list, col_list)), shape=(N, N))
    closure_csr.sum_duplicates()
    
    return closure_csr
```

#### Alternative: More Efficient Implementation

The above is illustrative. A more efficient approach:

```python
def _sparse_closure_topological_v2(graph: csr_matrix) -> csr_matrix:
    """
    More efficient implementation using LIL (List of Lists) format.
    """
    N = graph.shape[0]
    
    # Step 1: Topological sort
    topo_order = topological_sort(graph)
    
    # Step 2: Convert to LIL for efficient row operations
    closure = graph.tolil()
    
    # Step 3: Forward pass in topological order
    for v in topo_order:
        # Get predecessors (vertices that have edges TO v)
        predecessors = closure.getcol(v).nonzero()[0]
        
        # Get successors of v as dictionary {col: val}
        v_row = {col: val for col, val in zip(closure.rows[v], closure.data[v])}
        
        # For each predecessor u
        for u in predecessors:
            u_to_v = closure[u, v]
            
            # Add/update u->w for all w in successors of v
            for w, v_to_w in v_row.items():
                new_val = avos_product(u_to_v, v_to_w)
                existing = closure[u, w]
                
                if existing == 0:
                    # Add new edge
                    closure[u, w] = new_val
                else:
                    # Combine with existing edge
                    closure[u, w] = avos_sum(existing, new_val)
    
    # Convert back to CSR for efficiency
    return closure.tocsr()
```

### Phase 2: Cycle Detection and Fallback

```python
def sparse_transitive_closure(graph, method='auto', **kwargs):
    """Main entry point with method selection."""
    N = graph.shape[0]
    
    # Use dense algorithm for small graphs
    if N < kwargs.get('dense_threshold', 1000):
        return dense_transitive_closure(graph)
    
    if method == 'auto':
        # Check if graph is DAG
        try:
            topological_sort(graph)
            method = 'topological'
        except CycleError:
            method = 'warshall'
    
    if method == 'topological':
        return _sparse_closure_topological(graph)
    elif method == 'warshall':
        return _sparse_closure_warshall(graph)
    elif method == 'iterative':
        return _sparse_closure_iterative(graph)
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Phase 3: Integration with Existing API

Update `redblackgraph/sparse/csgraph/transitive_closure.py`:

```python
def transitive_closure(R, method='auto', directed=True, overwrite=False):
    """
    Compute transitive closure of a graph.
    
    Automatically uses sparse or dense algorithm based on graph size.
    """
    # Existing validation
    validate_graph(R, directed, ...)
    
    N = R.shape[0]
    
    # NEW: Use sparse algorithm for large graphs
    if N > 1000 and issparse(R):
        try:
            W = sparse_transitive_closure(
                R, 
                method=method, 
                overwrite=overwrite
            )
            # Convert to rbm_matrix
            return TransitiveClosure(W, None)
        except (CycleError, MemoryError) as e:
            # Fall back to dense if sparse fails
            warnings.warn(f"Sparse closure failed: {e}. Using dense algorithm.")
    
    # Existing dense implementation
    dist_matrix, predecessors = shortest_path(
        R, method=method, directed=True, overwrite=False
    )
    return TransitiveClosure(dist_matrix, predecessors)
```

## Implementation Plan

### Phase 1: Proof of Concept (Python)
- [ ] Implement topological sort for AVOS graphs
- [ ] Implement sparse closure using LIL format (Python)
- [ ] Add cycle detection
- [ ] Write unit tests with small graphs
- [ ] Benchmark against dense algorithm

### Phase 2: Optimization (Cython)
- [ ] Port topological sort to Cython
- [ ] Port sparse closure core loop to Cython
- [ ] Optimize AVOS operations for sparse case
- [ ] Add memory usage tracking
- [ ] Profile and optimize hot paths

### Phase 3: Integration
- [ ] Update `transitive_closure()` API
- [ ] Add automatic method selection
- [ ] Update `graph_builder.py` to remove workaround
- [ ] Add integration tests with large graphs
- [ ] Update documentation

### Phase 4: Advanced Features
- [ ] Implement Warshall's algorithm for cyclic graphs
- [ ] Add progress callbacks for long-running closures
- [ ] Add incremental closure updates
- [ ] Optimize for specific graph patterns (e.g., trees, chains)

## Testing Strategy

### Unit Tests

```python
def test_sparse_closure_small_dag():
    """Test correctness on small known DAG."""
    # Graph: 0->1, 1->2, 0->2
    # Closure: 0->1, 1->2, 0->2 (no new edges)
    pass

def test_sparse_closure_transitive():
    """Test that transitive edges are added."""
    # Graph: 0->1, 1->2
    # Closure: 0->1, 1->2, 0->2 (adds 0->2)
    pass

def test_sparse_closure_diamond():
    """Test diamond graph."""
    # Graph: 0->1, 0->2, 1->3, 2->3
    # Closure: all pairs (0,1), (0,2), (0,3), (1,3), (2,3)
    pass

def test_sparse_closure_avos_values():
    """Test that AVOS values are correctly computed."""
    # Use colored edges, verify AVOS products
    pass

def test_sparse_closure_cycle_detection():
    """Test that cycles are detected."""
    # Graph: 0->1, 1->2, 2->0
    # Should raise CycleError with method='topological'
    pass
```

### Integration Tests

```python
def test_sparse_vs_dense_equivalence():
    """Verify sparse and dense algorithms produce same results."""
    # Generate random DAGs
    # Compare outputs
    pass

def test_large_sparse_graph():
    """Test with large sparse graph that would fail with dense."""
    # N=100,000, nnz=10,000
    # Verify completes without memory error
    pass

def test_genealogy_graph():
    """Test with realistic genealogy graph structure."""
    # Create multi-generation family tree
    # Verify closure includes all ancestor relationships
    pass
```

### Performance Benchmarks

```python
def benchmark_sparse_vs_dense():
    """Compare performance of sparse vs dense algorithms."""
    # Vary: N, nnz, depth, branching factor
    # Measure: time, peak memory, output size
    pass
```

## Performance Expectations

### Small Graphs (N < 1000)
- Dense algorithm should be faster (simpler, cache-friendly)
- Continue using existing implementation

### Medium Graphs (1K < N < 100K, nnz < 1M)
- Sparse algorithm should be 10-100x faster
- Memory savings: 100-1000x

### Large Graphs (N > 100K, nnz < 10M)
- Dense algorithm: impossible (memory)
- Sparse algorithm: feasible (minutes to hours)
- Memory usage: proportional to nnz_closure

## Success Criteria

- [ ] Can compute transitive closure for N=1M vertex graph with nnz=10M
- [ ] Sparse algorithm is faster than dense for nnz < N²/100
- [ ] Memory usage is O(nnz_closure) not O(N²)
- [ ] All existing tests pass
- [ ] No regression in performance for small graphs (N < 1000)
- [ ] `graph_builder.py` workaround can be removed
- [ ] Integration with canonical ordering works for large graphs

## Future Enhancements

### Parallel Processing
- Parallelize topological sort pass
- Process multiple vertices in parallel when no dependencies
- Use multi-threading or MPI for very large graphs

### External Memory Algorithms
- For graphs where even nnz_closure doesn't fit in RAM
- Use disk-based storage with memory-mapped arrays
- Process in chunks/tiles

### Approximate Closures
- For extremely large graphs, compute approximate closure
- Sample-based reachability
- Useful for visualization or initial analysis

### Incremental Updates
- Efficiently update closure when edges are added/removed
- Useful for interactive applications
- Maintain closure as graph evolves

## References

- Warshall's Algorithm for transitive closure
- Topological sorting algorithms (Kahn's, DFS-based)
- Sparse matrix algorithms in SciPy
- AVOS algebra: redblackgraph documentation
- Related work: sparse matrix multiplication, graph traversal algorithms
