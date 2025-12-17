import numpy as np
from scipy.sparse import csr_matrix, isspmatrix

from redblackgraph.sparse import rb_matrix
from redblackgraph.types.transitive_closure import TransitiveClosure
from redblackgraph.sparse.csgraph._shortest_path import shortest_path, floyd_warshall
from redblackgraph.sparse.csgraph._components import (
    find_components_sparse, 
    get_component_vertices,
    extract_submatrix,
    merge_component_matrices
)
from redblackgraph.sparse.csgraph._topological_sort import is_upper_triangular, topological_sort
from redblackgraph.sparse.csgraph._density import DensificationError


def transitive_closure(R: rb_matrix, method="D", assume_upper_triangular=False) -> TransitiveClosure:
    """
    Compute the transitive closure of a red-black graph.
    
    Parameters
    ----------
    R : rb_matrix
        The input relationship matrix.
    method : str, default "D"
        Algorithm to use: "D" for Dijkstra, "FW" for Floyd-Warshall.
    assume_upper_triangular : bool, default False
        If True and method="FW", use optimized upper triangular algorithm.
        Use this after applying topological_sort to the graph.
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result.
    """
    return TransitiveClosure(*shortest_path(
        R, method=method, directed=True, overwrite=False,
        assume_upper_triangular=assume_upper_triangular
    ))


def transitive_closure_floyd_warshall(R: rb_matrix, assume_upper_triangular=False) -> TransitiveClosure:
    """
    Compute transitive closure using Floyd-Warshall algorithm.
    
    Parameters
    ----------
    R : rb_matrix
        The input relationship matrix.
    assume_upper_triangular : bool, default False
        If True, use optimized upper triangular algorithm for ~1.8-2x speedup.
        Use this after applying topological_sort to the graph.
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result.
    """
    return transitive_closure(R, method="FW", assume_upper_triangular=assume_upper_triangular)


def transitive_closure_dijkstra(R: rb_matrix) -> TransitiveClosure:
    """
    Compute transitive closure using Dijkstra's algorithm.
    
    Parameters
    ----------
    R : rb_matrix
        The input relationship matrix.
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result.
    """
    return transitive_closure(R, method="D")


def component_wise_closure(
    A,
    method: str = "auto",
    densify_threshold: int = 500
) -> TransitiveClosure:
    """
    Compute transitive closure by processing each connected component separately.
    
    This is memory-efficient for graphs with multiple disconnected components,
    as it avoids allocating a full N×N dense matrix. Each component is processed
    independently, and the results are merged into a sparse matrix.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input adjacency matrix
    method : str, default "auto"
        Algorithm for closure within each component:
        - "auto": Use FW for small components, Dijkstra for large
        - "FW": Always use Floyd-Warshall
        - "D": Always use Dijkstra
    densify_threshold : int, default 500
        For "auto" method, components smaller than this use Floyd-Warshall
        (which requires densification), larger ones use Dijkstra (sparse).
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result with sparse matrix W
        
    Notes
    -----
    Memory savings come from:
    1. Only allocating dense matrices for individual components, not full graph
    2. The merged result is sparse (no edges between components)
    
    For a graph with k equal-sized components, memory is O(N²/k) instead of O(N²).
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> # Two disconnected components
    >>> A = csr_matrix([[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]])
    >>> result = component_wise_closure(A)
    >>> # Result is sparse - no edges between components
    """
    # Ensure sparse format
    if not isspmatrix(A):
        A = csr_matrix(A)
    elif not isinstance(A, csr_matrix):
        A = A.tocsr()
    
    n = A.shape[0]
    
    if n == 0:
        return TransitiveClosure(csr_matrix((0, 0), dtype=np.int32), 0)
    
    # Find components
    q = {}
    component_labels = find_components_sparse(A, q)
    component_vertices = get_component_vertices(component_labels)
    
    n_components = len(component_vertices)
    
    # If single component, just use standard closure but ensure sparse output
    if n_components == 1:
        result = transitive_closure(A, method="FW" if method == "FW" else "D")
        if isspmatrix(result.W):
            return result
        else:
            return TransitiveClosure(csr_matrix(result.W), result.diameter)
    
    # Process each component
    closed_components = []
    max_diameter = 0
    
    for comp_id, vertices in component_vertices.items():
        comp_size = len(vertices)
        
        # Extract submatrix for this component
        submatrix, mapping = extract_submatrix(A, vertices)
        
        # Choose method based on size
        if method == "auto":
            if comp_size <= densify_threshold:
                comp_method = "FW"
            else:
                comp_method = "D"
        else:
            comp_method = method
        
        # Compute closure on component
        if comp_method == "FW":
            # Floyd-Warshall (densifies the component)
            from redblackgraph.core.redblack import array as rb_array
            dense_sub = rb_array(submatrix.toarray())
            closed_sub, diameter = floyd_warshall(dense_sub)
            closed_sub = csr_matrix(closed_sub)
        else:
            # Dijkstra (stays sparse)
            result = transitive_closure(submatrix, method="D")
            closed_sub = csr_matrix(result.W)
            diameter = result.diameter
        
        max_diameter = max(max_diameter, diameter)
        closed_components.append((closed_sub, mapping))
    
    # Merge all components back
    merged = merge_component_matrices(closed_components, n)
    
    return TransitiveClosure(merged, max_diameter)


def transitive_closure_squaring(A, max_iterations: int = 64) -> TransitiveClosure:
    """
    Compute transitive closure via repeated squaring.
    
    Uses the identity: TC(A) = A + A² + A⁴ + A⁸ + ...
    Converges in O(log d) iterations where d is the diameter.
    
    This method stays sparse throughout, making it suitable for very sparse graphs
    where Floyd-Warshall's O(N³) dense computation would be prohibitive.
    
    Parameters
    ----------
    A : sparse matrix
        Input adjacency matrix. Will be converted to rb_matrix for AVOS operations.
    max_iterations : int, default 64
        Maximum number of squaring iterations (prevents infinite loops for
        graphs with diameter > 2^64, which is essentially impossible).
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result with sparse matrix W.
        
    Notes
    -----
    Complexity: O(nnz² * log(d)) where nnz is number of non-zeros and d is diameter.
    For very sparse graphs this beats Floyd-Warshall's O(N³).
    
    The algorithm:
    1. Start with R = A (current reach)
    2. Compute R² using AVOS matmul
    3. Merge: R = R ⊕ R² (AVOS sum = min)
    4. If R unchanged, we've converged
    5. Repeat with R²
    """
    # Convert to rb_matrix for AVOS operations
    if not isinstance(A, rb_matrix):
        R = rb_matrix(A)
    else:
        R = A.copy()
    
    n = R.shape[0]
    if n == 0:
        return TransitiveClosure(csr_matrix((0, 0), dtype=np.int32), 0)
    
    # Repeated squaring
    for iteration in range(max_iterations):
        # Compute R² using AVOS matmul
        R_squared = R @ R
        
        # Merge R with R²: take element-wise min (AVOS sum)
        # For sparse matrices, we need to handle this carefully
        R_new = _sparse_avos_sum(R, R_squared)
        
        # Check convergence: if R unchanged, we're done
        if _sparse_equal(R, R_new):
            break
        
        R = R_new
    
    # Compute diameter from max value
    if R.nnz > 0:
        max_val = np.max(np.abs(R.data))
        diameter = int(np.floor(np.log2(max_val))) if max_val > 1 else 0
    else:
        diameter = 0
    
    return TransitiveClosure(R, diameter)


def _sparse_avos_sum(A, B):
    """
    Compute element-wise AVOS sum (min of non-zeros) of two sparse matrices.
    
    AVOS sum: a ⊕ b = min(a, b) where 0 is treated as infinity.
    """
    # Convert to rb_matrix if needed
    if not isinstance(A, rb_matrix):
        A = rb_matrix(A)
    if not isinstance(B, rb_matrix):
        B = rb_matrix(B)
    
    # For AVOS sum, we need element-wise min where 0 means "no edge"
    # A value is kept if it's non-zero; among non-zero values, keep the smaller
    
    # Get the union of sparsity patterns
    # Using lil_matrix for efficient construction
    from scipy.sparse import lil_matrix
    
    n = A.shape[0]
    result = lil_matrix((n, n), dtype=np.int32)
    
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    
    # Process A's entries
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            val_a = A_csr.data[idx]
            result[i, j] = val_a
    
    # Process B's entries, taking min where overlap
    for i in range(n):
        for idx in range(B_csr.indptr[i], B_csr.indptr[i + 1]):
            j = B_csr.indices[idx]
            val_b = B_csr.data[idx]
            
            current = result[i, j]
            if current == 0:
                result[i, j] = val_b
            else:
                # AVOS sum: min of absolute values, preserving sign
                # For RBG, we compare unsigned magnitudes
                if abs(val_b) < abs(current):
                    result[i, j] = val_b
    
    return rb_matrix(result.tocsr())


def _sparse_equal(A, B):
    """Check if two sparse matrices are equal."""
    if A.shape != B.shape:
        return False
    if A.nnz != B.nnz:
        return False
    
    diff = A - B
    return diff.nnz == 0


def transitive_closure_dag_sparse(A) -> TransitiveClosure:
    """
    Compute transitive closure of a DAG using truly sparse operations.
    
    This algorithm never allocates O(N²) memory. It uses topological ordering
    and propagates closure information from successors to predecessors.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input adjacency matrix. Must be a directed acyclic graph (DAG).
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result with sparse matrix W.
        
    Raises
    ------
    CycleError
        If the graph contains a cycle.
        
    Notes
    -----
    Algorithm:
    1. Compute topological ordering of vertices
    2. Process vertices in reverse topological order (sinks first)
    3. For each vertex v, its closure is: direct edges + union of successor closures
    4. Store closure for each vertex as a sparse row
    
    Complexity:
    - Time: O(V + E + nnz_closure) where nnz_closure is output non-zeros
    - Space: O(nnz_closure) - never allocates N×N dense matrix
    
    This is the only transitive closure algorithm in this module that
    guarantees no O(N²) memory allocation.
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> # Simple DAG: 0 -> 1 -> 2
    >>> A = csr_matrix([[1, 2, 0], [0, 1, 4], [0, 0, 1]], dtype=np.int32)
    >>> result = transitive_closure_dag_sparse(A)
    >>> # Closure adds edge 0 -> 2
    """
    from redblackgraph.sparse.csgraph.cycleerror import CycleError
    from redblackgraph.reference.rbg_math import avos_sum, avos_product
    
    # Convert to CSR if needed
    if not isspmatrix(A):
        A_csr = csr_matrix(A, dtype=np.int32)
    elif not isinstance(A, csr_matrix):
        A_csr = A.tocsr()
    else:
        A_csr = A
    
    n = A_csr.shape[0]
    
    if n == 0:
        return TransitiveClosure(csr_matrix((0, 0), dtype=np.int32), 0)
    
    # Get topological ordering (raises CycleError if graph has cycles)
    try:
        topo_order = topological_sort(A_csr)
    except CycleError as e:
        # Re-raise with the original vertex information
        raise CycleError(
            "transitive_closure_dag_sparse requires a DAG (no cycles)",
            vertex=e.vertex
        )
    
    # Process vertices in reverse topological order (sinks first)
    # For each vertex, we compute its closure row
    # closure[v] = {v: identity} ∪ {direct edges from v} ∪ {closure[w] for w in successors(v)}
    
    # Store closure for each vertex as a dictionary: col -> value
    # This is memory-efficient as we only store non-zero entries
    closure_rows = [None] * n  # closure_rows[v] = dict mapping col -> value
    
    # Get CSR arrays for efficient iteration
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    
    max_value = 0
    
    # Process in reverse topological order
    for v in reversed(topo_order):
        # Initialize closure for v with its direct edges
        v_closure = {}
        
        # Add direct edges from v (including self-loop/identity)
        for idx in range(indptr[v], indptr[v + 1]):
            col = indices[idx]
            val = data[idx]
            v_closure[col] = val
            if abs(val) > max_value:
                max_value = abs(val)
        
        # For each direct successor w of v, add w's closure to v's closure
        for idx in range(indptr[v], indptr[v + 1]):
            w = indices[idx]
            v_to_w = data[idx]
            
            # Skip self-loops for propagation
            if w == v:
                continue
            
            # Get w's closure (already computed since we process in reverse topo order)
            w_closure = closure_rows[w]
            if w_closure is None:
                continue
            
            # For each entry (w, x) -> val in w's closure, add (v, x) -> v_to_w ⊗ val
            for x, w_to_x in w_closure.items():
                # Compute AVOS product: v -> w -> x
                v_to_x = avos_product(v_to_w, w_to_x)
                
                if v_to_x == 0:
                    continue
                
                # AVOS sum with existing value (if any)
                if x in v_closure:
                    v_closure[x] = avos_sum(v_closure[x], v_to_x)
                else:
                    v_closure[x] = v_to_x
                
                if abs(v_closure[x]) > max_value:
                    max_value = abs(v_closure[x])
        
        closure_rows[v] = v_closure
    
    # Build CSR matrix from closure_rows
    # Count total non-zeros
    total_nnz = sum(len(row) if row else 0 for row in closure_rows)
    
    # Allocate arrays
    new_indptr = np.zeros(n + 1, dtype=np.int32)
    new_indices = np.zeros(total_nnz, dtype=np.int32)
    new_data = np.zeros(total_nnz, dtype=np.int32)
    
    # Fill arrays
    idx = 0
    for v in range(n):
        new_indptr[v] = idx
        v_closure = closure_rows[v]
        if v_closure:
            # Sort by column index for CSR format
            for col in sorted(v_closure.keys()):
                new_indices[idx] = col
                new_data[idx] = v_closure[col]
                idx += 1
    new_indptr[n] = idx
    
    # Create result matrix
    result = rb_matrix((new_data, new_indices, new_indptr), shape=(n, n))
    
    # Compute diameter from max value
    if max_value > 1:
        diameter = int(np.floor(np.log2(max_value)))
    else:
        diameter = 0
    
    return TransitiveClosure(result, diameter)


def transitive_closure_adaptive(
    A,
    method: str = "auto",
    density_threshold: float = 0.1,
    component_threshold: int = 1,
    size_threshold: int = 1000,
    sparse_only: bool = False
) -> TransitiveClosure:
    """
    Compute transitive closure using automatically selected optimal strategy.
    
    This function analyzes the input graph and selects the best algorithm:
    - Multiple components → component_wise_closure (process each separately)
    - Very sparse + large → Dijkstra (O(V * E * log V))
    - Upper triangular → Floyd-Warshall with optimization (~2x speedup)
    - Otherwise → Floyd-Warshall (accepts O(N³) densification)
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input adjacency matrix
    method : str, default "auto"
        Override algorithm selection:
        - "auto": Automatic selection based on graph properties
        - "FW": Force Floyd-Warshall
        - "D": Force Dijkstra
        - "component": Force component-wise processing
        - "squaring": Force repeated squaring
        - "dag_sparse": Force sparse DAG closure (no O(N²) allocation)
    density_threshold : float, default 0.1
        Graphs sparser than this use Dijkstra instead of FW
    component_threshold : int, default 1
        Use component-wise processing if more components than this
    size_threshold : int, default 1000
        Graphs larger than this prefer sparse algorithms
    sparse_only : bool, default False
        If True, only use algorithms that never allocate O(N²) memory.
        Currently this means using transitive_closure_dag_sparse, which
        requires the input to be a DAG. Raises DensificationError if the
        graph contains cycles or would otherwise require densification.
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result.
        
    Raises
    ------
    DensificationError
        If sparse_only=True and the graph would require O(N²) allocation
        (e.g., contains cycles).
        
    Notes
    -----
    Decision logic (when sparse_only=False):
    1. If multiple disconnected components → component_wise_closure
    2. Else if very sparse (< density_threshold) and large → Dijkstra
    3. Else if upper triangular → FW with assume_upper_triangular=True
    4. Else → standard Floyd-Warshall
    
    When sparse_only=True:
    - Uses transitive_closure_dag_sparse which never allocates O(N²) memory
    - Requires input to be a DAG (raises CycleError/DensificationError otherwise)
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([[1, 2, 0], [0, 1, 4], [0, 0, 1]])
    >>> result = transitive_closure_adaptive(A)
    >>> # With sparse_only mode (for large DAGs)
    >>> result = transitive_closure_adaptive(A, sparse_only=True)
    """
    from redblackgraph.sparse.csgraph.cycleerror import CycleError
    
    # Convert to sparse if needed
    if not isspmatrix(A):
        A_sparse = csr_matrix(A)
    else:
        A_sparse = A.tocsr() if not isinstance(A, csr_matrix) else A
    
    n = A_sparse.shape[0]
    
    # Handle empty matrix
    if n == 0:
        return TransitiveClosure(csr_matrix((0, 0), dtype=np.int32), 0)
    
    # Sparse-only mode: use DAG sparse closure
    if sparse_only:
        try:
            return transitive_closure_dag_sparse(A_sparse)
        except CycleError as e:
            raise DensificationError(
                f"sparse_only=True requires a DAG, but graph contains cycles: {e}",
                density=1.0,
                threshold=0.0
            )
    
    # Override methods
    if method == "FW":
        return transitive_closure(A_sparse, method="FW")
    elif method == "D":
        return transitive_closure(A_sparse, method="D")
    elif method == "component":
        return component_wise_closure(A_sparse)
    elif method == "squaring":
        return transitive_closure_squaring(A_sparse)
    elif method == "dag_sparse":
        return transitive_closure_dag_sparse(A_sparse)
    
    # Auto selection logic
    
    # 1. Check for multiple components
    q = {}
    component_labels = find_components_sparse(A_sparse, q)
    n_components = len(q)
    
    if n_components > component_threshold:
        # Multiple components - process separately for memory efficiency
        return component_wise_closure(A_sparse)
    
    # 2. Check density
    nnz = A_sparse.nnz
    density = nnz / (n * n) if n > 0 else 0
    
    if density < density_threshold and n > size_threshold:
        # Very sparse and large - use Dijkstra
        return transitive_closure(A_sparse, method="D")
    
    # 3. Check if upper triangular
    if is_upper_triangular(A_sparse):
        # Already upper triangular - use optimized FW
        from redblackgraph.core.redblack import array as rb_array
        if isspmatrix(A_sparse):
            A_dense = rb_array(A_sparse.toarray())
        else:
            A_dense = rb_array(np.asarray(A_sparse))
        result, diameter = floyd_warshall(A_dense, assume_upper_triangular=True)
        return TransitiveClosure(csr_matrix(result), diameter)
    
    # 4. Default: standard Floyd-Warshall
    return transitive_closure(A_sparse, method="FW")
