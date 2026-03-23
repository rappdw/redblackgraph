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


# Dynamic array for storing sparse row entries, with open-addressing hash map
# for O(1) column lookup instead of O(n) linear scan.
cdef struct SparseRow:
    ITYPE_t* cols      # Column indices (flat array)
    DTYPE_t* vals      # Values (flat array)
    ITYPE_t size       # Current number of entries
    ITYPE_t capacity   # Allocated capacity for cols/vals
    ITYPE_t* ht_keys   # Hash table keys (column indices, -1 = empty)
    ITYPE_t* ht_vals   # Hash table values (index into cols/vals)
    ITYPE_t ht_mask    # Hash table size - 1 (size is always power of 2)


# Sentinel for empty hash table slots
DEF HT_EMPTY = -1
# Minimum hash table size (must be power of 2)
DEF HT_MIN_SIZE = 16


cdef inline void sparse_row_init(SparseRow* row, ITYPE_t initial_capacity):
    """Initialize a sparse row with given capacity and hash table."""
    row.cols = <ITYPE_t*>malloc(initial_capacity * sizeof(ITYPE_t))
    row.vals = <DTYPE_t*>malloc(initial_capacity * sizeof(DTYPE_t))
    row.size = 0
    row.capacity = initial_capacity
    # Initialize hash table (size = HT_MIN_SIZE, load factor threshold ~50%)
    row.ht_mask = HT_MIN_SIZE - 1
    row.ht_keys = <ITYPE_t*>malloc(HT_MIN_SIZE * sizeof(ITYPE_t))
    row.ht_vals = <ITYPE_t*>malloc(HT_MIN_SIZE * sizeof(ITYPE_t))
    cdef ITYPE_t j
    for j in range(HT_MIN_SIZE):
        row.ht_keys[j] = HT_EMPTY


cdef inline void sparse_row_free(SparseRow* row):
    """Free a sparse row's memory."""
    if row.cols != NULL:
        free(row.cols)
        row.cols = NULL
    if row.vals != NULL:
        free(row.vals)
        row.vals = NULL
    if row.ht_keys != NULL:
        free(row.ht_keys)
        row.ht_keys = NULL
    if row.ht_vals != NULL:
        free(row.ht_vals)
        row.ht_vals = NULL
    row.size = 0
    row.capacity = 0
    row.ht_mask = 0


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


cdef inline ITYPE_t _ht_hash(ITYPE_t col, ITYPE_t mask):
    """Hash a column index. Multiply-shift hash for good distribution."""
    # Knuth multiplicative hash (golden ratio constant for 32-bit)
    cdef unsigned int h = <unsigned int>col * <unsigned int>2654435761u
    return <ITYPE_t>(h & <unsigned int>mask)


cdef inline void _ht_rehash(SparseRow* row):
    """Double the hash table size and reinsert all entries."""
    cdef ITYPE_t old_size = row.ht_mask + 1
    cdef ITYPE_t new_size = old_size * 2
    cdef ITYPE_t new_mask = new_size - 1
    cdef ITYPE_t* new_keys = <ITYPE_t*>malloc(new_size * sizeof(ITYPE_t))
    cdef ITYPE_t* new_vals = <ITYPE_t*>malloc(new_size * sizeof(ITYPE_t))
    cdef ITYPE_t j, slot, key

    for j in range(new_size):
        new_keys[j] = HT_EMPTY

    # Reinsert all existing entries
    for j in range(old_size):
        key = row.ht_keys[j]
        if key != HT_EMPTY:
            slot = _ht_hash(key, new_mask)
            while new_keys[slot] != HT_EMPTY:
                slot = (slot + 1) & new_mask
            new_keys[slot] = key
            new_vals[slot] = row.ht_vals[j]

    free(row.ht_keys)
    free(row.ht_vals)
    row.ht_keys = new_keys
    row.ht_vals = new_vals
    row.ht_mask = new_mask


cdef inline void sparse_row_add(SparseRow* row, ITYPE_t col, DTYPE_t val):
    """Add an entry to the sparse row (assumes col is not already present)."""
    cdef ITYPE_t pos = row.size
    sparse_row_ensure_capacity(row, pos + 1)
    row.cols[pos] = col
    row.vals[pos] = val
    row.size = pos + 1

    # Insert into hash table
    # Check load factor first: rehash if size > 50% of table capacity
    if row.size * 2 > row.ht_mask + 1:
        _ht_rehash(row)

    cdef ITYPE_t slot = _ht_hash(col, row.ht_mask)
    while row.ht_keys[slot] != HT_EMPTY:
        slot = (slot + 1) & row.ht_mask
    row.ht_keys[slot] = col
    row.ht_vals[slot] = pos


cdef inline ITYPE_t sparse_row_find(SparseRow* row, ITYPE_t col):
    """Find index of column in row, or -1 if not found. O(1) via hash table."""
    cdef ITYPE_t slot = _ht_hash(col, row.ht_mask)
    cdef ITYPE_t key
    while True:
        key = row.ht_keys[slot]
        if key == HT_EMPTY:
            return -1
        if key == col:
            return row.ht_vals[slot]
        slot = (slot + 1) & row.ht_mask


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
