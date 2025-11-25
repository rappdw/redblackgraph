import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix, isspmatrix_csr

from redblackgraph.core.redblack import array as rb_array
from typing import Dict, Optional, Sequence, List, Tuple

include 'parameters.pxi'
include '_rbg_math.pxi'
include '_csr_utils.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
def find_components(A: rb_array, q:Optional[Dict[int, int]] = None) -> Sequence[int]:
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (this implementation assumes that it is transitively closed)
    :param q: if set, should be defaultdict(lambda: 0)
    :return: a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
    """

    # Component identification is usually done using iterative dfs for each vertex. Since A is
    # transitively closed, we have implicit DFS info in each row. This algorithm utilizes that
    # fact. Conceptually, this algorithm "crawls" the matrix.
    #
    # This is our algorithm:
    #
    # Allocate an array that will represent the component for each vertex
    # Allocate an array that will represent the vertices that have been visited
    # Iterate over each vertex that hasn't been visited:
    #   This is a new component, so increment the component id
    #   Allocate a set to hold ids to be added to this component
    #   Add the current vertex to this set
    #   while the set is not empty
    #     pull a vextex from the set
    #     add it to the current component
    #     For each non-zero cell in the vertex's row and column add those vertices to the set for this component

    cdef unsigned int n = len(A)
    vertices = range(n)
    component_for_vertex_np = np.zeros((n), dtype=np.uint32)
    cdef unsigned int[ : ] component_for_vertex = component_for_vertex_np
    cdef unsigned char[ : ] visited_vertices = np.zeros((n), dtype=np.uint8)
    cdef unsigned int component_id = 0
    cdef unsigned int vertex_count
    cdef DTYPE_t[:, :] Am = A
    for i in vertices: # it.filterfalse(lambda x: visited_vertices[x], vertices):
        if visited_vertices[i]:
            continue
        vertices_added_to_component = set()
        vertex_count = 0
        vertices_added_to_component.add(i)
        while vertices_added_to_component:
            vertex = vertices_added_to_component.pop()
            vertex_count += 1
            visited_vertices[vertex] = True
            component_for_vertex[vertex] = component_id
            for j in vertices:
                if not ((Am[vertex][j] == 0 and Am[j][vertex] == 0) or visited_vertices[j] or j in vertices_added_to_component):
                    vertices_added_to_component.add(j)
        if q is not None:
            q[component_id] = vertex_count
        component_id += 1
    return component_for_vertex_np


@cython.boundscheck(False)
@cython.wraparound(False)
def find_components_sparse(A, q: Optional[Dict[int, int]] = None) -> np.ndarray:
    """
    Find connected components in a sparse matrix using O(V+E) time.
    
    This is the sparse-optimized version that uses CSR/CSC iteration
    instead of O(n²) dense iteration.
    
    Parameters
    ----------
    A : sparse matrix
        Input adjacency matrix (will be converted to CSR if needed)
    q : dict, optional
        If provided, will be populated with component_id -> vertex_count
        
    Returns
    -------
    np.ndarray
        Array of length n where component_for_vertex[i] = component_id
    """
    # Ensure CSR format
    if not isspmatrix_csr(A):
        A_csr = A.tocsr() if isspmatrix(A) else csr_matrix(A)
    else:
        A_csr = A
    
    # Also need CSC for incoming edges (column access)
    A_csc = A_csr.tocsc()
    
    cdef ITYPE_t n = A_csr.shape[0]
    
    # Get CSR arrays
    cdef np.ndarray[ITYPE_t, ndim=1] csr_indptr = np.asarray(A_csr.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] csr_indices = np.asarray(A_csr.indices, dtype=ITYPE)
    
    # Get CSC arrays
    cdef np.ndarray[ITYPE_t, ndim=1] csc_indptr = np.asarray(A_csc.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] csc_indices = np.asarray(A_csc.indices, dtype=ITYPE)
    
    # Output arrays
    cdef np.ndarray[UDTYPE_t, ndim=1] component_for_vertex = np.zeros(n, dtype=UDTYPE)
    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    
    # Typed memoryviews for fast access
    cdef ITYPE_t[:] csr_indptr_v = csr_indptr
    cdef ITYPE_t[:] csr_indices_v = csr_indices
    cdef ITYPE_t[:] csc_indptr_v = csc_indptr
    cdef ITYPE_t[:] csc_indices_v = csc_indices
    cdef UDTYPE_t[:] comp_v = component_for_vertex
    cdef np.uint8_t[:] visited_v = visited
    
    cdef UDTYPE_t component_id = 0
    cdef ITYPE_t i, vertex, neighbor, idx
    cdef UDTYPE_t vertex_count
    
    # Stack for iterative DFS (avoids Python set overhead in inner loop)
    cdef list stack
    
    for i in range(n):
        if visited_v[i]:
            continue
        
        # Start new component
        stack = [i]
        vertex_count = 0
        
        while stack:
            vertex = stack.pop()
            
            if visited_v[vertex]:
                continue
            
            visited_v[vertex] = 1
            comp_v[vertex] = component_id
            vertex_count += 1
            
            # Outgoing edges (row iteration in CSR) - O(out_degree)
            for idx in range(csr_indptr_v[vertex], csr_indptr_v[vertex + 1]):
                neighbor = csr_indices_v[idx]
                if not visited_v[neighbor]:
                    stack.append(neighbor)
            
            # Incoming edges (column iteration in CSC) - O(in_degree)
            for idx in range(csc_indptr_v[vertex], csc_indptr_v[vertex + 1]):
                neighbor = csc_indices_v[idx]
                if not visited_v[neighbor]:
                    stack.append(neighbor)
        
        if q is not None:
            q[component_id] = vertex_count
        component_id += 1
    
    return component_for_vertex


def extract_submatrix(A, vertices: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    """
    Extract a submatrix for a given set of vertices.
    
    This extracts the subgraph induced by the specified vertices,
    returning a new sparse matrix with vertices renumbered 0..len(vertices)-1.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix in CSR format (or convertible to CSR)
    vertices : np.ndarray
        Sorted array of vertex indices to extract
        
    Returns
    -------
    tuple of (csr_matrix, np.ndarray)
        - Extracted submatrix
        - Mapping array: new_to_old[new_idx] = old_idx
    """
    if not isspmatrix_csr(A):
        A = A.tocsr() if isspmatrix(A) else csr_matrix(A)
    
    cdef ITYPE_t n_full = A.shape[0]
    cdef ITYPE_t n_sub = len(vertices)
    
    # Build old_to_new mapping (-1 means not in subset)
    cdef np.ndarray[ITYPE_t, ndim=1] old_to_new = np.full(n_full, -1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] vertices_arr = np.asarray(vertices, dtype=ITYPE)
    
    cdef ITYPE_t i
    for i in range(n_sub):
        old_to_new[vertices_arr[i]] = i
    
    # Get CSR arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_in = np.asarray(A.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_in = np.asarray(A.indices, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_in = np.asarray(A.data, dtype=DTYPE)
    
    # Count non-zeros in submatrix
    cdef ITYPE_t nnz_sub = 0
    cdef ITYPE_t v, idx, col, new_col
    
    for i in range(n_sub):
        v = vertices_arr[i]
        for idx in range(indptr_in[v], indptr_in[v + 1]):
            col = indices_in[idx]
            if old_to_new[col] >= 0:
                nnz_sub += 1
    
    # Allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_out = np.empty(n_sub + 1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_out = np.empty(nnz_sub, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_out = np.empty(nnz_sub, dtype=DTYPE)
    
    # Fill output arrays
    cdef ITYPE_t out_idx = 0
    indptr_out[0] = 0
    
    for i in range(n_sub):
        v = vertices_arr[i]
        for idx in range(indptr_in[v], indptr_in[v + 1]):
            col = indices_in[idx]
            new_col = old_to_new[col]
            if new_col >= 0:
                indices_out[out_idx] = new_col
                data_out[out_idx] = data_in[idx]
                out_idx += 1
        indptr_out[i + 1] = out_idx
    
    result = csr_matrix((data_out, indices_out, indptr_out), shape=(n_sub, n_sub))
    return result, vertices_arr


def merge_component_matrices(
    components: List[Tuple[csr_matrix, np.ndarray]],
    n_total: int
) -> csr_matrix:
    """
    Merge multiple component submatrices into a full sparse matrix.
    
    This reconstructs a full matrix from component submatrices extracted
    via extract_submatrix.
    
    Parameters
    ----------
    components : list of (csr_matrix, np.ndarray)
        List of (submatrix, new_to_old_mapping) tuples
    n_total : int
        Total number of vertices in full matrix
        
    Returns
    -------
    csr_matrix
        Merged sparse matrix of shape (n_total, n_total)
    """
    # Calculate total nnz
    cdef ITYPE_t total_nnz = sum(comp[0].nnz for comp in components)
    
    # Allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_out = np.zeros(n_total + 1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_out = np.empty(total_nnz, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_out = np.empty(total_nnz, dtype=DTYPE)
    
    # First pass: count entries per row
    cdef ITYPE_t i, old_row, idx
    cdef np.ndarray[ITYPE_t, ndim=1] mapping
    
    for submat, mapping in components:
        indptr_sub = np.asarray(submat.indptr, dtype=ITYPE)
        n_sub = len(mapping)
        for i in range(n_sub):
            old_row = mapping[i]
            row_nnz = indptr_sub[i + 1] - indptr_sub[i]
            indptr_out[old_row + 1] = row_nnz
    
    # Cumulative sum for indptr
    for i in range(n_total):
        indptr_out[i + 1] += indptr_out[i]
    
    # Track current position in each row
    cdef np.ndarray[ITYPE_t, ndim=1] row_pos = indptr_out[:-1].copy()
    
    # Second pass: fill data
    for submat, mapping in components:
        indptr_sub = np.asarray(submat.indptr, dtype=ITYPE)
        indices_sub = np.asarray(submat.indices, dtype=ITYPE)
        data_sub = np.asarray(submat.data, dtype=DTYPE)
        n_sub = len(mapping)
        
        for i in range(n_sub):
            old_row = mapping[i]
            for idx in range(indptr_sub[i], indptr_sub[i + 1]):
                new_col_in_sub = indices_sub[idx]
                old_col = mapping[new_col_in_sub]
                
                pos = row_pos[old_row]
                indices_out[pos] = old_col
                data_out[pos] = data_sub[idx]
                row_pos[old_row] += 1
    
    result = csr_matrix((data_out, indices_out, indptr_out), shape=(n_total, n_total))
    result.sort_indices()
    return result


def get_component_vertices(component_labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Get vertices for each component from component labels.
    
    Parameters
    ----------
    component_labels : np.ndarray
        Array where component_labels[i] = component_id for vertex i
        
    Returns
    -------
    dict
        Dictionary mapping component_id -> array of vertex indices
    """
    unique_labels = np.unique(component_labels)
    return {
        int(label): np.where(component_labels == label)[0].astype(ITYPE)
        for label in unique_labels
    }
