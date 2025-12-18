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
    if not isspmatrix(A):
        A_sparse = csr_matrix(A)
    else:
        A_sparse = A.tocsr() if not isinstance(A, csr_matrix) else A
    
    n = A_sparse.shape[0]
    if n == 0:
        return TransitiveClosure(csr_matrix((0, 0), dtype=np.int32), 0)
    
    if is_upper_triangular(A_sparse):
        return transitive_closure_dag_sparse(A_sparse)
    
    R = rb_matrix(A_sparse) if not isinstance(A_sparse, rb_matrix) else A_sparse.copy()
    
    for iteration in range(max_iterations):
        R_squared = R @ R
        R_squared.eliminate_zeros()
        R_new = _sparse_avos_sum(R, R_squared)
        
        if _sparse_equal(R, R_new):
            break
        
        R = R_new
    
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
    
    Uses efficient COO-based merge instead of element-wise lil_matrix access.
    """
    from scipy.sparse import coo_matrix
    
    # Convert to COO for efficient concatenation
    A_coo = A.tocoo()
    B_coo = B.tocoo()
    
    n = A.shape[0]
    
    # Handle empty matrices
    if A_coo.nnz == 0 and B_coo.nnz == 0:
        return rb_matrix(csr_matrix((n, n), dtype=np.int32))
    if A_coo.nnz == 0:
        result = rb_matrix(B_coo.tocsr())
        result.eliminate_zeros()
        return result
    if B_coo.nnz == 0:
        result = rb_matrix(A_coo.tocsr())
        result.eliminate_zeros()
        return result
    
    # Concatenate all entries from A and B
    all_rows = np.concatenate([A_coo.row, B_coo.row])
    all_cols = np.concatenate([A_coo.col, B_coo.col])
    all_data = np.concatenate([A_coo.data, B_coo.data])
    
    # Sort by (row, col) to group duplicates together
    # Use lexsort: sorts by last key first, so (col, row) gives row-major order
    sort_idx = np.lexsort((all_cols, all_rows))
    sorted_rows = all_rows[sort_idx]
    sorted_cols = all_cols[sort_idx]
    sorted_data = all_data[sort_idx]
    
    # Find unique (row, col) positions and reduce duplicates using AVOS sum
    # AVOS sum: min of absolute values, preserving sign of the minimum
    # If abs values tie, keep the first one (from A)
    
    # Create composite key for finding unique positions
    composite_key = sorted_rows.astype(np.int64) * n + sorted_cols.astype(np.int64)
    
    # Find where keys change (boundaries between groups)
    key_changes = np.concatenate([[True], composite_key[1:] != composite_key[:-1]])
    
    # For each group, we need to find the element with minimum absolute value
    # Use a simple approach: iterate through groups
    unique_indices = np.where(key_changes)[0]
    n_unique = len(unique_indices)
    
    result_rows = np.empty(n_unique, dtype=np.int32)
    result_cols = np.empty(n_unique, dtype=np.int32)
    result_data = np.empty(n_unique, dtype=np.int32)
    
    # Process each group
    for i in range(n_unique):
        start = unique_indices[i]
        end = unique_indices[i + 1] if i + 1 < n_unique else len(sorted_data)
        
        result_rows[i] = sorted_rows[start]
        result_cols[i] = sorted_cols[start]
        
        # Find minimum absolute value in this group
        group_data = sorted_data[start:end]
        abs_vals = np.abs(group_data)
        min_idx = np.argmin(abs_vals)
        result_data[i] = group_data[min_idx]
    
    # Filter out zeros (AVOS: 0 means no edge)
    nonzero_mask = result_data != 0
    result_rows = result_rows[nonzero_mask]
    result_cols = result_cols[nonzero_mask]
    result_data = result_data[nonzero_mask]
    
    # Build result CSR matrix
    result_coo = coo_matrix((result_data, (result_rows, result_cols)), shape=(n, n), dtype=np.int32)
    result = rb_matrix(result_coo.tocsr())
    result.sum_duplicates()
    result.sort_indices()
    
    return result


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
    
    This is the sparse-matrix-optimized version of the algorithm. For a pure
    Python reference implementation that works with plain lists, see
    :func:`redblackgraph.reference.transitive_closure_dag`.
    
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
    
    See Also
    --------
    redblackgraph.reference.transitive_closure_dag : Pure Python reference implementation
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> # Simple DAG: 0 -> 1 -> 2
    >>> A = csr_matrix([[1, 2, 0], [0, 1, 4], [0, 0, 1]], dtype=np.int32)
    >>> result = transitive_closure_dag_sparse(A)
    >>> # Closure adds edge 0 -> 2
    """
    # Use the Cython implementation for performance
    from ._transitive_closure_dag import transitive_closure_dag_sparse_cython
    return transitive_closure_dag_sparse_cython(A)


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
