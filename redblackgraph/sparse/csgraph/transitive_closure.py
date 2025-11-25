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