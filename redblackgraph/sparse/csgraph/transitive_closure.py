from redblackgraph.sparse import rb_matrix
from redblackgraph.types.transitive_closure import TransitiveClosure
from redblackgraph.sparse.csgraph._shortest_path import shortest_path


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