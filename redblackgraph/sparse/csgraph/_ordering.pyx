import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import isspmatrix, isspmatrix_csr, csr_matrix

from typing import Dict, List, Sequence
from redblackgraph.types.ordering import Ordering
from ._components import find_components
from ._permutation import permute, permute_sparse

include 'parameters.pxi'
include '_rbg_math.pxi'
include '_csr_utils.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_permutation(A: Sequence[Sequence[int]], q: Dict[int, int], ids: Sequence[int]) -> List[int]:
    # This is the default sort ordering used by Traingularization
    # it sorts by:
    #   * size of component, descending
    #   * component id, ascending
    #   * relationship count, descending
    #   * max ancestor: ascending
    #   * color: descending
    #   * vertex_id: ascending
    cdef unsigned int n = len(A)
    cdef DTYPE_t[:] max_rel_for_vertex = np.zeros((n), dtype=np.int32)
    cdef DTYPE_t[:] ancester_count_for_vertex = np.zeros((n), dtype=np.int32)
    cdef DTYPE_t[:, :] Am = A
    vertices = range(n)
    for i in vertices:
        for j in vertices:
            if Am[i][j]:
                ancester_count_for_vertex[i] += MSB(Am[i][j])
            if Am[j][i]:
                max_rel_for_vertex[i] = max(max_rel_for_vertex[i], Am[j][i])

    basis = [i for i in range(len(ids))]
    # sort descending on size of component and "ancestor count", ascending on all other elements
    basis.sort(key=lambda x: (-q[ids[x]], ids[x], -ancester_count_for_vertex[x], max_rel_for_vertex[x], -A[x][x], x))
    return basis


def _get_permutation_sparse(A, q: Dict[int, int], ids: np.ndarray) -> List[int]:
    """
    Sparse version of _get_permutation using O(nnz) CSR/CSC iteration.
    
    This computes the same canonical ordering as _get_permutation but uses
    sparse matrix iteration patterns for O(V+E) complexity instead of O(V²).
    
    Parameters
    ----------
    A : sparse matrix
        Input adjacency matrix in CSR format (or convertible)
    q : dict
        Component sizes: component_id -> vertex_count
    ids : np.ndarray
        Component ID for each vertex
        
    Returns
    -------
    list
        Permutation array for canonical ordering
    """
    # Ensure CSR format
    if not isspmatrix_csr(A):
        A_csr = A.tocsr() if isspmatrix(A) else csr_matrix(A)
    else:
        A_csr = A
    
    # Also need CSC for column iteration (max_rel_for_vertex)
    A_csc = A_csr.tocsc()
    
    n = A_csr.shape[0]
    
    # Compute ancestor counts and max relationships using helper
    ancestor_count, max_rel, diagonal = _compute_ordering_metrics_sparse(
        A_csr.indptr, A_csr.indices, A_csr.data,
        A_csc.indptr, A_csc.indices, A_csc.data,
        n
    )
    
    # Build permutation using same sort key as dense version
    basis = list(range(n))
    ids_arr = np.asarray(ids)
    
    # Sort by: (-component_size, component_id, -ancestor_count, max_rel, -diagonal, vertex_id)
    basis.sort(key=lambda x: (
        -q[ids_arr[x]], 
        ids_arr[x], 
        -ancestor_count[x], 
        max_rel[x], 
        -diagonal[x], 
        x
    ))
    return basis


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _compute_ordering_metrics_sparse(
    csr_indptr, csr_indices, csr_data,
    csc_indptr, csc_indices, csc_data,
    ITYPE_t n
):
    """
    Compute ancestor counts, max relationships, and diagonal values.
    
    This is a cdef helper to allow typed arrays.
    """
    # Get typed arrays
    cdef np.ndarray[ITYPE_t, ndim=1] csr_indptr_arr = np.asarray(csr_indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] csr_indices_arr = np.asarray(csr_indices, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] csr_data_arr = np.asarray(csr_data, dtype=DTYPE)
    
    cdef np.ndarray[ITYPE_t, ndim=1] csc_indptr_arr = np.asarray(csc_indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] csc_indices_arr = np.asarray(csc_indices, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] csc_data_arr = np.asarray(csc_data, dtype=DTYPE)
    
    # Output arrays
    cdef np.ndarray[DTYPE_t, ndim=1] ancestor_count = np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] max_rel = np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] diagonal = np.zeros(n, dtype=DTYPE)
    
    cdef ITYPE_t i, idx, col, row
    cdef DTYPE_t value
    
    # Ancestor counts from rows - O(nnz) not O(n²)
    # ancestor_count[i] = sum of MSB of all non-zero values in row i
    for i in range(n):
        for idx in range(csr_indptr_arr[i], csr_indptr_arr[i + 1]):
            col = csr_indices_arr[idx]
            value = csr_data_arr[idx]
            if value != 0:
                ancestor_count[i] += MSB(value)
            # Also capture diagonal
            if col == i:
                diagonal[i] = value
    
    # Max relationships from columns - O(nnz) not O(n²)
    # max_rel[i] = max value in column i (i.e., max of A[j][i] for all j)
    for i in range(n):
        for idx in range(csc_indptr_arr[i], csc_indptr_arr[i + 1]):
            row = csc_indices_arr[idx]
            value = csc_data_arr[idx]
            if value != 0:
                max_rel[i] = max(max_rel[i], value)
    
    return ancestor_count, max_rel, diagonal


def avos_canonical_ordering(A) -> Ordering:
    """
    Canonically sort the matrix.
    
    Automatically dispatches to sparse or dense implementation based on input type.

    This ordering is canonical. Graph components will appear in adjacent
    rows starting with the largest component in rows 0-n, the next largest in n+1-m, etc.
    Should the graph hold components of the same size, the component id will be used to
    order one above the other. Within a component, row ordering is determined first by
    maximum relationship value in a row and finally by original vertex id.

    :param A: input matrix (assumed to be transitively closed)
    :return: an upper triangular matrix that is isomorphic to A
    """
    q = dict()  # dictionary keyed by component id, value is count of vertices in component
    components = find_components(A, q)
    
    if isspmatrix(A):
        # Sparse path - O(V+E) complexity
        permutation = np.array(_get_permutation_sparse(A, q, components), dtype=ITYPE)
        return Ordering(permute_sparse(A, permutation), permutation, q)
    else:
        # Dense path - original O(V²) implementation
        permutation = np.array(_get_permutation(A, q, components), dtype=ITYPE)
        return Ordering(permute(A, permutation), permutation, q)
