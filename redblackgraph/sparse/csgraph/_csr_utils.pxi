# CSR Iteration Primitives for Sparse Operations
# This is a Cython include file (.pxi) providing inline functions for efficient
# CSR/CSC sparse matrix traversal in O(nnz) time instead of O(n²).

cimport numpy as np
from libc.stdlib cimport malloc, free

# =============================================================================
# Row Edge Iteration (CSR format)
# =============================================================================

cdef inline ITYPE_t csr_row_start(ITYPE_t[:] indptr, ITYPE_t row) noexcept nogil:
    """Get the starting index in indices/data arrays for a given row."""
    return indptr[row]

cdef inline ITYPE_t csr_row_end(ITYPE_t[:] indptr, ITYPE_t row) noexcept nogil:
    """Get the ending index (exclusive) in indices/data arrays for a given row."""
    return indptr[row + 1]

cdef inline ITYPE_t csr_row_nnz(ITYPE_t[:] indptr, ITYPE_t row) noexcept nogil:
    """Get the number of non-zero entries in a given row."""
    return indptr[row + 1] - indptr[row]

cdef inline void iterate_row_edges(
    ITYPE_t[:] indptr,
    ITYPE_t[:] indices,
    DTYPE_t[:] data,
    ITYPE_t row,
    ITYPE_t* out_cols,
    DTYPE_t* out_vals,
    ITYPE_t* out_count
) noexcept nogil:
    """
    Iterate over non-zero edges in a CSR row.
    
    Parameters
    ----------
    indptr : array of row pointers
    indices : array of column indices
    data : array of values
    row : row index to iterate
    out_cols : output array for column indices (must be pre-allocated)
    out_vals : output array for values (must be pre-allocated)
    out_count : output count of edges found
    
    Usage:
        cdef ITYPE_t cols[MAX_DEGREE]
        cdef DTYPE_t vals[MAX_DEGREE]
        cdef ITYPE_t count
        iterate_row_edges(indptr, indices, data, row, cols, vals, &count)
    """
    cdef ITYPE_t start = indptr[row]
    cdef ITYPE_t end = indptr[row + 1]
    cdef ITYPE_t i, idx = 0
    
    for i in range(start, end):
        out_cols[idx] = indices[i]
        out_vals[idx] = data[i]
        idx += 1
    
    out_count[0] = idx


# =============================================================================
# Column Edge Iteration (requires CSC format or transpose)
# =============================================================================

cdef inline ITYPE_t csc_col_start(ITYPE_t[:] indptr, ITYPE_t col) noexcept nogil:
    """Get the starting index in indices/data arrays for a given column (CSC format)."""
    return indptr[col]

cdef inline ITYPE_t csc_col_end(ITYPE_t[:] indptr, ITYPE_t col) noexcept nogil:
    """Get the ending index (exclusive) in indices/data arrays for a given column (CSC format)."""
    return indptr[col + 1]

cdef inline ITYPE_t csc_col_nnz(ITYPE_t[:] indptr, ITYPE_t col) noexcept nogil:
    """Get the number of non-zero entries in a given column (CSC format)."""
    return indptr[col + 1] - indptr[col]


# =============================================================================
# CSR Transpose (build CSC from CSR in O(nnz))
# =============================================================================

cdef inline void build_csr_transpose(
    ITYPE_t n_rows,
    ITYPE_t n_cols,
    ITYPE_t nnz,
    ITYPE_t[:] indptr_in,
    ITYPE_t[:] indices_in,
    DTYPE_t[:] data_in,
    ITYPE_t[:] indptr_out,
    ITYPE_t[:] indices_out,
    DTYPE_t[:] data_out
) noexcept nogil:
    """
    Build CSR transpose (equivalent to CSC) in O(nnz) time.
    
    Given CSR(A), builds CSR(A^T) which is equivalent to CSC(A).
    
    Parameters
    ----------
    n_rows : number of rows in input
    n_cols : number of columns in input (rows in output)
    nnz : number of non-zeros
    indptr_in : input row pointers
    indices_in : input column indices
    data_in : input values
    indptr_out : output row pointers (size n_cols + 1, pre-allocated)
    indices_out : output column indices (size nnz, pre-allocated)
    data_out : output values (size nnz, pre-allocated)
    
    Algorithm:
    1. Count entries per output row (column counts of input)
    2. Compute cumulative sum to get indptr_out
    3. Place entries in output arrays
    """
    cdef ITYPE_t i, j, idx, col, dest
    cdef ITYPE_t* col_counts
    
    # Initialize output indptr to zeros
    for i in range(n_cols + 1):
        indptr_out[i] = 0
    
    # Pass 1: Count entries per column (which becomes rows in transpose)
    for i in range(nnz):
        col = indices_in[i]
        indptr_out[col + 1] += 1
    
    # Compute cumulative sum
    for i in range(n_cols):
        indptr_out[i + 1] += indptr_out[i]
    
    # Allocate workspace for tracking current position in each output row
    col_counts = <ITYPE_t*>malloc(n_cols * sizeof(ITYPE_t))
    for i in range(n_cols):
        col_counts[i] = 0
    
    # Pass 2: Place entries
    for i in range(n_rows):
        for idx in range(indptr_in[i], indptr_in[i + 1]):
            col = indices_in[idx]
            dest = indptr_out[col] + col_counts[col]
            indices_out[dest] = i  # Row becomes column in transpose
            data_out[dest] = data_in[idx]
            col_counts[col] += 1
    
    free(col_counts)


# =============================================================================
# Sparse Permutation Utilities
# =============================================================================

cdef inline void compute_inverse_permutation(
    ITYPE_t[:] p,
    ITYPE_t[:] p_inv,
    ITYPE_t n
) noexcept nogil:
    """
    Compute inverse permutation: p_inv[p[i]] = i
    
    Parameters
    ----------
    p : permutation array
    p_inv : output inverse permutation (pre-allocated, size n)
    n : size of permutation
    """
    cdef ITYPE_t i
    for i in range(n):
        p_inv[p[i]] = i


cdef inline void permute_csr_rows(
    ITYPE_t n,
    ITYPE_t nnz,
    ITYPE_t[:] indptr_in,
    ITYPE_t[:] indices_in,
    DTYPE_t[:] data_in,
    ITYPE_t[:] p,
    ITYPE_t[:] p_inv,
    ITYPE_t[:] indptr_out,
    ITYPE_t[:] indices_out,
    DTYPE_t[:] data_out
) noexcept nogil:
    """
    Permute a CSR matrix: B = P * A * P^T
    
    This reorders both rows and columns according to permutation p.
    Output row i contains input row p[i], with columns remapped via p_inv.
    
    Parameters
    ----------
    n : matrix dimension
    nnz : number of non-zeros
    indptr_in, indices_in, data_in : input CSR arrays
    p : permutation (new_idx -> old_idx)
    p_inv : inverse permutation (old_idx -> new_idx)
    indptr_out, indices_out, data_out : output CSR arrays (pre-allocated)
    
    Note: Output arrays must be pre-allocated:
        indptr_out: size n + 1
        indices_out: size nnz
        data_out: size nnz
    """
    cdef ITYPE_t i, j, old_row, old_col, new_col, idx_in, idx_out
    cdef ITYPE_t row_start, row_end, row_nnz
    
    # Build indptr_out by counting entries per output row
    indptr_out[0] = 0
    for i in range(n):
        old_row = p[i]
        row_nnz = indptr_in[old_row + 1] - indptr_in[old_row]
        indptr_out[i + 1] = indptr_out[i] + row_nnz
    
    # Copy entries with column remapping
    idx_out = 0
    for i in range(n):
        old_row = p[i]
        row_start = indptr_in[old_row]
        row_end = indptr_in[old_row + 1]
        
        for idx_in in range(row_start, row_end):
            old_col = indices_in[idx_in]
            new_col = p_inv[old_col]
            indices_out[idx_out] = new_col
            data_out[idx_out] = data_in[idx_in]
            idx_out += 1


# =============================================================================
# Component/Submatrix Extraction Utilities
# =============================================================================

cdef inline ITYPE_t count_submatrix_nnz(
    ITYPE_t[:] indptr,
    ITYPE_t[:] indices,
    ITYPE_t[:] vertex_mask,
    ITYPE_t[:] vertices,
    ITYPE_t n_vertices
) noexcept nogil:
    """
    Count non-zeros in a submatrix defined by a vertex set.
    
    Parameters
    ----------
    indptr, indices : CSR arrays of full matrix
    vertex_mask : boolean mask (1 if vertex is in subset, 0 otherwise)
    vertices : list of vertices in subset
    n_vertices : number of vertices in subset
    
    Returns
    -------
    Number of non-zeros in the submatrix
    """
    cdef ITYPE_t count = 0
    cdef ITYPE_t i, v, idx, col
    
    for i in range(n_vertices):
        v = vertices[i]
        for idx in range(indptr[v], indptr[v + 1]):
            col = indices[idx]
            if vertex_mask[col]:
                count += 1
    
    return count


cdef inline void extract_submatrix_csr(
    ITYPE_t[:] indptr_in,
    ITYPE_t[:] indices_in,
    DTYPE_t[:] data_in,
    ITYPE_t[:] vertices,
    ITYPE_t n_vertices,
    ITYPE_t[:] old_to_new,
    ITYPE_t[:] indptr_out,
    ITYPE_t[:] indices_out,
    DTYPE_t[:] data_out
) noexcept nogil:
    """
    Extract a submatrix from CSR format.
    
    Parameters
    ----------
    indptr_in, indices_in, data_in : input CSR arrays
    vertices : sorted list of vertex indices to extract
    n_vertices : number of vertices
    old_to_new : mapping from old vertex index to new (size of full matrix)
    indptr_out, indices_out, data_out : output CSR arrays (pre-allocated)
    """
    cdef ITYPE_t i, v, idx, col, new_col
    cdef ITYPE_t out_idx = 0
    
    indptr_out[0] = 0
    
    for i in range(n_vertices):
        v = vertices[i]
        for idx in range(indptr_in[v], indptr_in[v + 1]):
            col = indices_in[idx]
            new_col = old_to_new[col]
            if new_col >= 0:  # -1 means not in subset
                indices_out[out_idx] = new_col
                data_out[out_idx] = data_in[idx]
                out_idx += 1
        indptr_out[i + 1] = out_idx
