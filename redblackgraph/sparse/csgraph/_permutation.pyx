import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csr

include 'parameters.pxi'
include '_rbg_math.pxi'
include '_csr_utils.pxi'


def permute(A, p, assume_upper_triangular=False):
    """
    Permute a matrix based on the vertex ordering specified.
    
    Dispatches to sparse or dense implementation based on input type.
    Equivalent to P * A * P^-1 (where P is a permutation matrix specified by p)
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix to permute
    p : array-like
        Permutation array: p[new_idx] = old_idx
    assume_upper_triangular : bool, default False
        If True, only process upper triangular portion
        
    Returns
    -------
    array-like or sparse matrix
        Permuted matrix (same type as input)
    """
    if isspmatrix(A):
        return permute_sparse(A, p, assume_upper_triangular)
    else:
        return _permute_dense(A, p, assume_upper_triangular)


def _permute_dense(A, p, assume_upper_triangular=False):
    '''Permutes an input matrix based on the vertex ordering specified.

    Equivalent to P * A * P-1 (where P is a permutation of the identity matrix specified by p)
    '''
    cdef np.ndarray B = np.zeros(A.shape, dtype=DTYPE)
    _permute(A, B, p, assume_upper_triangular)
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t _permute(np.ndarray[DTYPE_t, ndim=2, mode='c'] A, np.ndarray[DTYPE_t, ndim=2, mode='c'] B, np.ndarray[ITYPE_t, ndim=1, mode='c'] p, bint assume_upper_triangular):
    cdef unsigned int i, j, start, N = B.shape[0]
    assert B.shape[1] == N
    assert p.shape[0] == N
    cdef DTYPE_t[:, :] Am = A
    cdef DTYPE_t[:, :] Bm = B
    cdef ITYPE_t[:] pm = p

    for i in range(N):
        if assume_upper_triangular:
            start = i
        else:
            start = 0

        for j in range(start, N):
            Bm[i][j] = Am[pm[i]][pm[j]]


def permute_sparse(A, p, assume_upper_triangular=False):
    """
    Permute a sparse matrix based on the vertex ordering specified.
    
    This is the sparse-preserving version that maintains O(nnz) complexity
    and does NOT densify the matrix.
    
    Equivalent to P * A * P^-1 (where P is a permutation matrix specified by p)
    
    Parameters
    ----------
    A : sparse matrix
        Input sparse matrix to permute (will be converted to CSR if needed)
    p : array-like
        Permutation array: p[new_idx] = old_idx
    assume_upper_triangular : bool, default False
        If True, filter to keep only upper triangular entries in output
        
    Returns
    -------
    csr_matrix
        Permuted sparse matrix in CSR format
    """
    # Ensure CSR format
    if not isspmatrix_csr(A):
        A = A.tocsr()
    
    cdef ITYPE_t n = A.shape[0]
    cdef ITYPE_t nnz = A.nnz
    
    # Convert permutation to numpy array if needed
    cdef np.ndarray[ITYPE_t, ndim=1] p_arr = np.asarray(p, dtype=ITYPE)
    
    # Compute inverse permutation: p_inv[old_idx] = new_idx
    cdef np.ndarray[ITYPE_t, ndim=1] p_inv = np.empty(n, dtype=ITYPE)
    cdef ITYPE_t[:] p_view = p_arr
    cdef ITYPE_t[:] p_inv_view = p_inv
    compute_inverse_permutation(p_view, p_inv_view, n)
    
    # Get CSR arrays from input
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_in = np.asarray(A.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_in = np.asarray(A.indices, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_in = np.asarray(A.data, dtype=DTYPE)
    
    if assume_upper_triangular:
        # Need to filter entries, count first
        return _permute_sparse_upper_tri(n, indptr_in, indices_in, data_in, p_arr, p_inv)
    else:
        # Full permutation - output has same nnz as input
        return _permute_sparse_full(n, nnz, indptr_in, indices_in, data_in, p_arr, p_inv)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _permute_sparse_full(
    ITYPE_t n,
    ITYPE_t nnz,
    np.ndarray[ITYPE_t, ndim=1] indptr_in,
    np.ndarray[ITYPE_t, ndim=1] indices_in,
    np.ndarray[DTYPE_t, ndim=1] data_in,
    np.ndarray[ITYPE_t, ndim=1] p,
    np.ndarray[ITYPE_t, ndim=1] p_inv
):
    """Internal: Full sparse permutation without filtering."""
    # Allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_out = np.empty(n + 1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_out = np.empty(nnz, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_out = np.empty(nnz, dtype=DTYPE)
    
    # Use the inline permutation function
    cdef ITYPE_t[:] indptr_in_v = indptr_in
    cdef ITYPE_t[:] indices_in_v = indices_in
    cdef DTYPE_t[:] data_in_v = data_in
    cdef ITYPE_t[:] p_v = p
    cdef ITYPE_t[:] p_inv_v = p_inv
    cdef ITYPE_t[:] indptr_out_v = indptr_out
    cdef ITYPE_t[:] indices_out_v = indices_out
    cdef DTYPE_t[:] data_out_v = data_out
    
    permute_csr_rows(n, nnz, indptr_in_v, indices_in_v, data_in_v,
                     p_v, p_inv_v, indptr_out_v, indices_out_v, data_out_v)
    
    # Sort indices within each row (CSR requirement)
    result = csr_matrix((data_out, indices_out, indptr_out), shape=(n, n))
    result.sort_indices()
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _permute_sparse_upper_tri(
    ITYPE_t n,
    np.ndarray[ITYPE_t, ndim=1] indptr_in,
    np.ndarray[ITYPE_t, ndim=1] indices_in,
    np.ndarray[DTYPE_t, ndim=1] data_in,
    np.ndarray[ITYPE_t, ndim=1] p,
    np.ndarray[ITYPE_t, ndim=1] p_inv
):
    """Internal: Sparse permutation keeping only upper triangular entries."""
    cdef ITYPE_t i, old_row, idx, old_col, new_col, out_idx
    cdef ITYPE_t row_start, row_end
    cdef DTYPE_t val
    
    # First pass: count upper triangular entries
    cdef ITYPE_t nnz_upper = 0
    for i in range(n):
        old_row = p[i]  # new row i comes from old row p[i]
        row_start = indptr_in[old_row]
        row_end = indptr_in[old_row + 1]
        for idx in range(row_start, row_end):
            old_col = indices_in[idx]
            new_col = p_inv[old_col]  # old column maps to new column
            if new_col >= i:  # upper triangular: col >= row
                nnz_upper += 1
    
    # Allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr_out = np.empty(n + 1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices_out = np.empty(nnz_upper, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data_out = np.empty(nnz_upper, dtype=DTYPE)
    
    # Second pass: fill arrays
    indptr_out[0] = 0
    out_idx = 0
    for i in range(n):
        old_row = p[i]
        row_start = indptr_in[old_row]
        row_end = indptr_in[old_row + 1]
        for idx in range(row_start, row_end):
            old_col = indices_in[idx]
            new_col = p_inv[old_col]
            if new_col >= i:  # upper triangular
                indices_out[out_idx] = new_col
                data_out[out_idx] = data_in[idx]
                out_idx += 1
        indptr_out[i + 1] = out_idx
    
    # Sort indices within each row
    result = csr_matrix((data_out, indices_out, indptr_out), shape=(n, n))
    result.sort_indices()
    return result