"""
Sparse Topological Sort for Red-Black Graphs

Implements O(V+E) topological sorting using iterative DFS with CSR iteration.
Produces a permutation that, when applied, yields an upper triangular matrix.
"""

import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import isspmatrix, isspmatrix_csr, csr_matrix

from redblackgraph.types.ordering import Ordering
from ._permutation import permute_sparse, permute
from ._components import find_components, find_components_sparse
from .cycleerror import CycleError

include 'parameters.pxi'
include '_csr_utils.pxi'


# DFS state constants
DEF WHITE = 0  # Not visited
DEF GRAY = 1   # Currently in recursion stack (being explored)
DEF BLACK = 2  # Finished processing


@cython.boundscheck(False)
@cython.wraparound(False)
def topological_sort(A):
    """
    Compute a topological ordering of a directed acyclic graph.
    
    Uses iterative DFS with explicit stack for O(V+E) complexity on sparse graphs.
    The result is a permutation array such that applying it to the matrix yields
    an upper triangular matrix (all edges go from lower to higher index).
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Adjacency matrix of a DAG. Must be a directed acyclic graph.
        
    Returns
    -------
    np.ndarray
        Permutation array p where p[new_idx] = old_idx.
        Applying this permutation produces topological order.
        
    Raises
    ------
    CycleError
        If the graph contains a cycle.
        
    Notes
    -----
    For sparse inputs, complexity is O(V+E) where V is vertices and E is edges.
    For dense inputs, the matrix is first converted to CSR format.
    
    The resulting permutation satisfies: for all edges (i,j) in original graph,
    the new indices satisfy new_i < new_j.
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> # Lower triangular matrix (already in reverse topological order)
    >>> A = csr_matrix([[1, 0, 0], [2, 1, 0], [4, 2, 1]], dtype=np.int32)
    >>> p = topological_sort(A)
    >>> # p will reorder to make upper triangular
    """
    # Convert to CSR if needed
    if not isspmatrix(A):
        A_csr = csr_matrix(np.asarray(A), dtype=DTYPE)
    elif not isspmatrix_csr(A):
        A_csr = A.tocsr()
    else:
        A_csr = A
    
    cdef ITYPE_t n = A_csr.shape[0]
    
    if n == 0:
        return np.array([], dtype=ITYPE)
    
    # Get CSR arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr = np.asarray(A_csr.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.asarray(A_csr.indices, dtype=ITYPE)
    
    # Result arrays
    cdef np.ndarray[ITYPE_t, ndim=1] order = np.empty(n, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] color = np.zeros(n, dtype=ITYPE)  # WHITE=0
    
    # Perform iterative DFS
    cdef ITYPE_t order_idx = n - 1  # Fill from end (reverse post-order)
    cdef ITYPE_t vertex, neighbor, idx, start, end
    cdef ITYPE_t stack_top
    cdef bint has_unvisited_neighbor
    
    # Stack entries: (vertex, next_neighbor_idx)
    # Using two parallel arrays for efficiency
    cdef np.ndarray[ITYPE_t, ndim=1] stack_vertex = np.empty(n, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] stack_next_idx = np.empty(n, dtype=ITYPE)
    
    cdef ITYPE_t[:] indptr_v = indptr
    cdef ITYPE_t[:] indices_v = indices
    cdef ITYPE_t[:] order_v = order
    cdef ITYPE_t[:] color_v = color
    cdef ITYPE_t[:] stack_vertex_v = stack_vertex
    cdef ITYPE_t[:] stack_next_idx_v = stack_next_idx
    
    # Start DFS from each unvisited vertex
    cdef ITYPE_t start_vertex
    for start_vertex in range(n):
        if color_v[start_vertex] != WHITE:
            continue
        
        # Initialize stack with start vertex
        stack_top = 0
        stack_vertex_v[0] = start_vertex
        stack_next_idx_v[0] = indptr_v[start_vertex]
        color_v[start_vertex] = GRAY
        
        while stack_top >= 0:
            vertex = stack_vertex_v[stack_top]
            idx = stack_next_idx_v[stack_top]
            end = indptr_v[vertex + 1]
            
            # Find next unvisited neighbor
            has_unvisited_neighbor = False
            while idx < end:
                neighbor = indices_v[idx]
                idx += 1
                
                # Skip self-loops (diagonal entries)
                if neighbor == vertex:
                    continue
                
                if color_v[neighbor] == WHITE:
                    # Found unvisited neighbor - save state and push neighbor
                    stack_next_idx_v[stack_top] = idx
                    
                    stack_top += 1
                    stack_vertex_v[stack_top] = neighbor
                    stack_next_idx_v[stack_top] = indptr_v[neighbor]
                    color_v[neighbor] = GRAY
                    
                    has_unvisited_neighbor = True
                    break
                    
                elif color_v[neighbor] == GRAY:
                    # Back edge detected - cycle!
                    raise CycleError("Graph contains a cycle", vertex=neighbor)
            
            if not has_unvisited_neighbor:
                # All neighbors processed - mark as finished
                color_v[vertex] = BLACK
                order_v[order_idx] = vertex
                order_idx -= 1
                stack_top -= 1
    
    # order[] now contains vertices in topological order (sources first)
    # We need to return a permutation p where p[new_idx] = old_idx
    # This means the first vertex in topological order gets new index 0, etc.
    # So the order array IS the permutation we want
    return order


def topological_ordering(A):
    """
    Compute a topological ordering and apply it to produce an upper triangular matrix.
    
    This is a convenience wrapper that:
    1. Computes the topological sort permutation
    2. Applies the permutation to the matrix
    3. Returns an Ordering object with the result
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Adjacency matrix of a DAG.
        
    Returns
    -------
    Ordering
        Named tuple containing:
        - A: The permuted matrix (upper triangular)
        - label_permutation: The permutation applied
        - components: Dictionary mapping component IDs to sizes
        
    Raises
    ------
    CycleError
        If the graph contains a cycle.
        
    Notes
    -----
    The result matrix will be upper triangular: all non-zero entries A[i,j]
    with i != j will have i < j.
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix([[1, 0, 0], [2, 1, 0], [4, 2, 1]], dtype=np.int32)
    >>> result = topological_ordering(A)
    >>> # result.A is upper triangular
    """
    # Get the permutation
    perm = topological_sort(A)
    
    # Find components for the Ordering object
    q = dict()
    if isspmatrix(A):
        components = find_components_sparse(A, q)
        permuted = permute_sparse(A, perm, assume_upper_triangular=True)
    else:
        components = find_components(A, q)
        permuted = permute(A, perm, assume_upper_triangular=True)
    
    return Ordering(permuted, perm, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def is_upper_triangular(A, bint strict=False):
    """
    Check if a matrix is upper triangular.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Matrix to check.
    strict : bool, default False
        If True, diagonal must be zero (strictly upper triangular).
        If False, diagonal can be non-zero.
        
    Returns
    -------
    bool
        True if the matrix is upper triangular.
    """
    if not isspmatrix(A):
        A_csr = csr_matrix(np.asarray(A), dtype=DTYPE)
    elif not isspmatrix_csr(A):
        A_csr = A.tocsr()
    else:
        A_csr = A
    
    cdef ITYPE_t n = A_csr.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1] indptr = np.asarray(A_csr.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.asarray(A_csr.indices, dtype=ITYPE)
    
    cdef ITYPE_t[:] indptr_v = indptr
    cdef ITYPE_t[:] indices_v = indices
    
    cdef ITYPE_t i, idx, col
    
    for i in range(n):
        for idx in range(indptr_v[i], indptr_v[i + 1]):
            col = indices_v[idx]
            if strict:
                if col <= i:
                    return False
            else:
                if col < i:
                    return False
    
    return True
