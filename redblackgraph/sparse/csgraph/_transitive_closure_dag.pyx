"""
Cython implementation of sparse DAG transitive closure.

This module provides an optimized Cython implementation of transitive closure
for directed acyclic graphs (DAGs) that never allocates O(N²) memory.
"""

import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import isspmatrix, isspmatrix_csr, csr_matrix
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy

from redblackgraph.types.transitive_closure import TransitiveClosure
from redblackgraph.sparse import rb_matrix
from .cycleerror import CycleError
from ._topological_sort import topological_sort

include 'parameters.pxi'
include '_rbg_math.pxi'


# Dynamic array for storing sparse row entries
cdef struct SparseRow:
    ITYPE_t* cols      # Column indices
    DTYPE_t* vals      # Values
    ITYPE_t size       # Current number of entries
    ITYPE_t capacity   # Allocated capacity


cdef inline void sparse_row_init(SparseRow* row, ITYPE_t initial_capacity):
    """Initialize a sparse row with given capacity."""
    row.cols = <ITYPE_t*>malloc(initial_capacity * sizeof(ITYPE_t))
    row.vals = <DTYPE_t*>malloc(initial_capacity * sizeof(DTYPE_t))
    row.size = 0
    row.capacity = initial_capacity


cdef inline void sparse_row_free(SparseRow* row):
    """Free a sparse row's memory."""
    if row.cols != NULL:
        free(row.cols)
        row.cols = NULL
    if row.vals != NULL:
        free(row.vals)
        row.vals = NULL
    row.size = 0
    row.capacity = 0


cdef inline void sparse_row_ensure_capacity(SparseRow* row, ITYPE_t needed):
    """Ensure the row has at least 'needed' capacity."""
    cdef ITYPE_t new_capacity
    if needed > row.capacity:
        new_capacity = row.capacity * 2
        if new_capacity < needed:
            new_capacity = needed
        row.cols = <ITYPE_t*>realloc(row.cols, new_capacity * sizeof(ITYPE_t))
        row.vals = <DTYPE_t*>realloc(row.vals, new_capacity * sizeof(DTYPE_t))
        row.capacity = new_capacity


cdef inline void sparse_row_add(SparseRow* row, ITYPE_t col, DTYPE_t val):
    """Add an entry to the sparse row (assumes col is not already present)."""
    sparse_row_ensure_capacity(row, row.size + 1)
    row.cols[row.size] = col
    row.vals[row.size] = val
    row.size += 1


cdef inline ITYPE_t sparse_row_find(SparseRow* row, ITYPE_t col):
    """Find index of column in row, or -1 if not found."""
    cdef ITYPE_t i
    for i in range(row.size):
        if row.cols[i] == col:
            return i
    return -1


cdef inline void sparse_row_set_or_add(SparseRow* row, ITYPE_t col, DTYPE_t val):
    """Set value at column, or add if not present. Uses AVOS sum for existing entries."""
    cdef ITYPE_t idx = sparse_row_find(row, col)
    if idx >= 0:
        row.vals[idx] = avos_sum(row.vals[idx], val)
    else:
        sparse_row_add(row, col, val)


@cython.boundscheck(False)
@cython.wraparound(False)
def transitive_closure_dag_sparse_cython(A):
    """
    Compute transitive closure of a DAG using truly sparse operations (Cython).
    
    This is the Cython-optimized version of the algorithm. For a pure Python
    reference implementation, see :func:`redblackgraph.reference.transitive_closure_dag`.
    
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
    
    See Also
    --------
    redblackgraph.reference.transitive_closure_dag : Pure Python reference implementation
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
        return TransitiveClosure(csr_matrix((0, 0), dtype=DTYPE), 0)
    
    # Get topological ordering (raises CycleError if graph has cycles)
    cdef np.ndarray[ITYPE_t, ndim=1] topo_order
    try:
        topo_order = topological_sort(A_csr)
    except CycleError as e:
        raise CycleError(
            "transitive_closure_dag_sparse requires a DAG (no cycles)",
            vertex=e.vertex
        )
    
    # Get CSR arrays
    cdef np.ndarray[ITYPE_t, ndim=1] indptr = np.asarray(A_csr.indptr, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.asarray(A_csr.indices, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] data = np.asarray(A_csr.data, dtype=DTYPE)
    
    cdef ITYPE_t[:] indptr_v = indptr
    cdef ITYPE_t[:] indices_v = indices
    cdef DTYPE_t[:] data_v = data
    cdef ITYPE_t[:] topo_v = topo_order
    
    # Allocate closure rows for each vertex
    cdef SparseRow* closure_rows = <SparseRow*>malloc(n * sizeof(SparseRow))
    cdef ITYPE_t i, v, w, x, idx, w_idx
    cdef DTYPE_t v_to_w, w_to_x, v_to_x, val
    cdef DTYPE_t max_value = 0
    
    # Initialize all closure rows
    for i in range(n):
        sparse_row_init(&closure_rows[i], 8)  # Start with small capacity
    
    # Process vertices in reverse topological order (sinks first)
    for i in range(n - 1, -1, -1):
        v = topo_v[i]
        
        # Add direct edges from v (including self-loop/identity)
        for idx in range(indptr_v[v], indptr_v[v + 1]):
            w = indices_v[idx]
            val = data_v[idx]
            sparse_row_add(&closure_rows[v], w, val)
            if val > max_value or -val > max_value:
                if val > 0:
                    max_value = val
                else:
                    max_value = -val
        
        # For each direct successor w of v, add w's closure to v's closure
        for idx in range(indptr_v[v], indptr_v[v + 1]):
            w = indices_v[idx]
            v_to_w = data_v[idx]
            
            # Skip self-loops for propagation
            if w == v:
                continue
            
            # For each entry in w's closure, add to v's closure
            for w_idx in range(closure_rows[w].size):
                x = closure_rows[w].cols[w_idx]
                w_to_x = closure_rows[w].vals[w_idx]
                
                # Compute AVOS product: v -> w -> x
                v_to_x = avos_product(v_to_w, w_to_x)
                
                if v_to_x == 0:
                    continue
                
                # AVOS sum with existing value (if any)
                sparse_row_set_or_add(&closure_rows[v], x, v_to_x)
                
                # Update max value
                if v_to_x > max_value or -v_to_x > max_value:
                    if v_to_x > 0:
                        max_value = v_to_x
                    else:
                        max_value = -v_to_x
    
    # Count total non-zeros
    cdef ITYPE_t total_nnz = 0
    for v in range(n):
        total_nnz += closure_rows[v].size
    
    # Build CSR arrays for result
    cdef np.ndarray[ITYPE_t, ndim=1] new_indptr = np.zeros(n + 1, dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=1] new_indices = np.zeros(total_nnz, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] new_data = np.zeros(total_nnz, dtype=DTYPE)
    
    cdef ITYPE_t[:] new_indptr_v = new_indptr
    cdef ITYPE_t[:] new_indices_v = new_indices
    cdef DTYPE_t[:] new_data_v = new_data
    
    # Fill arrays (need to sort each row by column index for CSR format)
    cdef ITYPE_t out_idx = 0
    cdef np.ndarray[ITYPE_t, ndim=1] sort_indices
    cdef ITYPE_t j, sorted_idx
    
    for v in range(n):
        new_indptr_v[v] = out_idx
        
        if closure_rows[v].size > 0:
            # Create numpy arrays from the sparse row for sorting
            row_cols = np.empty(closure_rows[v].size, dtype=ITYPE)
            row_vals = np.empty(closure_rows[v].size, dtype=DTYPE)
            for j in range(closure_rows[v].size):
                row_cols[j] = closure_rows[v].cols[j]
                row_vals[j] = closure_rows[v].vals[j]
            
            # Sort by column index
            sort_indices = np.argsort(row_cols).astype(ITYPE)
            
            for j in range(closure_rows[v].size):
                sorted_idx = sort_indices[j]
                new_indices_v[out_idx] = row_cols[sorted_idx]
                new_data_v[out_idx] = row_vals[sorted_idx]
                out_idx += 1
    
    new_indptr_v[n] = out_idx
    
    # Free closure rows
    for v in range(n):
        sparse_row_free(&closure_rows[v])
    free(closure_rows)
    
    # Create result matrix
    result = rb_matrix((new_data, new_indices, new_indptr), shape=(n, n))
    
    # Compute diameter from max value
    cdef int diameter = 0
    if max_value > 1:
        diameter = MSB(max_value)
    
    return TransitiveClosure(result, diameter)
