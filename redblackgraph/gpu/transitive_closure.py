"""
GPU-resident transitive closure algorithms.

Two paths:
1. Repeated squaring: TC(A) = A + A² + A⁴ + ... via AVOS SpGEMM.
   Works for any graph. O(nnz² · log d) per iteration.
2. DAG-specific: Level-parallel topological propagation.
   Only for DAGs (triangular matrices). O(V + E + nnz_closure).
   Processes vertices level-by-level, with full GPU parallelism within each level.
"""

import numpy as np
from typing import Tuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .csr_gpu import CSRMatrixGPU
from .spgemm import spgemm


def transitive_closure_gpu(
    A: CSRMatrixGPU,
    max_iterations: int = 64
) -> Tuple[CSRMatrixGPU, int]:
    """
    Compute transitive closure via repeated squaring on GPU.

    Uses the identity: TC(A) = A + A² + A⁴ + A⁸ + ...
    Converges in O(log d) iterations where d is the graph diameter.

    All data stays GPU-resident — no CPU transfers during the loop.

    Args:
        A: Input adjacency matrix on GPU
        max_iterations: Maximum squaring iterations (prevents runaway)

    Returns:
        Tuple of (closure matrix, diameter estimate)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU operations")

    if A.nnz == 0:
        return A.copy(), 0

    R = A
    diameter = 0

    for iteration in range(max_iterations):
        R_squared = spgemm(R, R, upper_triangular=R.triangular)
        R_new = sparse_avos_sum_gpu(R, R_squared)

        if sparse_equal_gpu(R, R_new):
            break

        R = R_new
        diameter = iteration + 1

    # Estimate diameter from max value
    if R.nnz > 0:
        data_cpu = R.data.get()
        # Filter out identity values (-1, 1) for diameter estimate
        abs_vals = np.abs(data_cpu)
        non_identity = abs_vals[abs_vals > 1]
        if len(non_identity) > 0:
            max_val = int(np.max(non_identity))
            diameter = max(diameter, int(np.floor(np.log2(max_val))) if max_val > 1 else 0)

    return R, diameter


def sparse_avos_sum_gpu(A: CSRMatrixGPU, B: CSRMatrixGPU) -> CSRMatrixGPU:
    """
    Element-wise AVOS sum (min of non-zeros) of two sparse GPU matrices.

    AVOS sum: a ⊕ b = min(a, b) where 0 is treated as infinity.

    For entries present in only one matrix, that value is kept.
    For entries present in both, the minimum absolute value is kept
    (with its original sign).

    All operations happen on GPU.

    Args:
        A: First CSR matrix on GPU
        B: Second CSR matrix on GPU

    Returns:
        Result CSR matrix with AVOS sum of A and B
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    n_rows, n_cols = A.shape

    # Handle empty matrices
    if A.nnz == 0 and B.nnz == 0:
        return A.copy()
    if A.nnz == 0:
        return B.copy()
    if B.nnz == 0:
        return A.copy()

    # Convert both to COO on GPU
    rows_A = _csr_to_row_indices(A.indptr, A.nnz)
    rows_B = _csr_to_row_indices(B.indptr, B.nnz)

    # Concatenate all entries
    all_rows = cp.concatenate([rows_A, rows_B])
    all_cols = cp.concatenate([A.indices, B.indices])
    all_data = cp.concatenate([A.data, B.data])

    # Sort by (row, col)
    sort_key = all_rows.astype(cp.int64) * n_cols + all_cols.astype(cp.int64)
    order = cp.argsort(sort_key)
    sorted_rows = all_rows[order]
    sorted_cols = all_cols[order]
    sorted_data = all_data[order]
    sorted_keys = sort_key[order]

    # Find unique (row, col) positions
    key_changes = cp.concatenate([cp.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    unique_indices = cp.where(key_changes)[0]

    # For each group of duplicates, compute AVOS sum (min of abs values)
    result_data = _reduce_avos_sum(sorted_data, unique_indices, len(all_data))
    result_rows = sorted_rows[unique_indices]
    result_cols = sorted_cols[unique_indices]

    # Filter out zeros
    nonzero_mask = result_data != 0
    result_rows = result_rows[nonzero_mask]
    result_cols = result_cols[nonzero_mask]
    result_data = result_data[nonzero_mask]

    # Build CSR from COO
    indptr = cp.zeros(n_rows + 1, dtype=cp.int32)
    if len(result_rows) > 0:
        cp.add.at(indptr[1:], result_rows, 1)
        indptr = cp.cumsum(indptr).astype(cp.int32)

    triangular = A.triangular and B.triangular

    return CSRMatrixGPU(
        result_data,
        result_cols.astype(cp.int32),
        indptr,
        (n_rows, n_cols),
        triangular=triangular,
        validate=False
    )


def sparse_equal_gpu(A: CSRMatrixGPU, B: CSRMatrixGPU) -> bool:
    """
    Check if two sparse GPU matrices are identical.

    Comparison happens entirely on GPU — only a single bool is transferred.

    Args:
        A: First CSR matrix
        B: Second CSR matrix

    Returns:
        True if matrices are identical
    """
    if A.shape != B.shape:
        return False
    if A.nnz != B.nnz:
        return False
    if A.nnz == 0:
        return True

    # Compare arrays element-wise on GPU
    return (
        bool(cp.array_equal(A.indptr, B.indptr))
        and bool(cp.array_equal(A.indices, B.indices))
        and bool(cp.array_equal(A.data, B.data))
    )


_CSR_TO_ROW_KERNEL = r'''
extern "C" __global__ void csr_to_row_indices(
    const int* __restrict__ indptr,
    int* __restrict__ row_indices,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    for (int i = start; i < end; i++) {
        row_indices[i] = row;
    }
}
'''

_csr_to_row_module = None


def _csr_to_row_indices(indptr: 'cp.ndarray', nnz: int) -> 'cp.ndarray':
    """Convert CSR indptr to row index array (COO row indices). All on GPU."""
    global _csr_to_row_module
    n_rows = len(indptr) - 1
    row_indices = cp.empty(nnz, dtype=cp.int32)
    if nnz == 0:
        return row_indices
    if _csr_to_row_module is None:
        _csr_to_row_module = cp.RawModule(code=_CSR_TO_ROW_KERNEL)
    kernel = _csr_to_row_module.get_function('csr_to_row_indices')
    block_size = 256
    grid_size = (n_rows + block_size - 1) // block_size
    kernel((grid_size,), (block_size,),
           (indptr, row_indices, n_rows))
    return row_indices


_REDUCE_AVOS_KERNEL = r'''
extern "C" __global__ void reduce_avos_sum(
    const int* __restrict__ sorted_data,
    const long long* __restrict__ unique_indices,
    const long long* __restrict__ group_ends,
    int* __restrict__ result,
    int n_unique
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_unique) return;

    long long start = unique_indices[tid];
    long long end = group_ends[tid];

    if (end - start == 1) {
        result[tid] = sorted_data[start];
        return;
    }

    // Find element with minimum absolute value
    int best = sorted_data[start];
    int best_abs = best < 0 ? -best : best;

    for (long long i = start + 1; i < end; i++) {
        int val = sorted_data[i];
        int val_abs = val < 0 ? -val : val;
        if (val_abs < best_abs) {
            best = val;
            best_abs = val_abs;
        }
    }

    result[tid] = best;
}
'''

_reduce_avos_module = None


def _get_reduce_avos_kernel():
    global _reduce_avos_module
    if _reduce_avos_module is None:
        _reduce_avos_module = cp.RawModule(code=_REDUCE_AVOS_KERNEL)
    return _reduce_avos_module.get_function('reduce_avos_sum')


def _reduce_avos_sum(
    sorted_data: 'cp.ndarray',
    unique_indices: 'cp.ndarray',
    total_len: int
) -> 'cp.ndarray':
    """
    For each group of duplicate (row, col) entries, compute AVOS sum.
    All on GPU via a CUDA kernel.
    """
    n_unique = len(unique_indices)
    result = cp.empty(n_unique, dtype=cp.int32)

    # Ensure int64 to match unique_indices (from cp.where)
    unique_indices = cp.asarray(unique_indices, dtype=cp.int64)
    group_ends = cp.concatenate([unique_indices[1:], cp.array([total_len], dtype=cp.int64)])

    block_size = 256
    grid_size = (n_unique + block_size - 1) // block_size
    kernel = _get_reduce_avos_kernel()
    kernel((grid_size,), (block_size,),
           (sorted_data, unique_indices, group_ends, result, n_unique))

    return result


# ---------------------------------------------------------------------------
# DAG-specific transitive closure: level-parallel topological propagation
# ---------------------------------------------------------------------------

_DAG_KERNELS_CODE = r'''
extern "C" {

__device__ inline int MSB(int x) {
    int bit_position = 0;
    while (x > 1) { x >>= 1; bit_position++; }
    return bit_position;
}

__device__ inline int avos_product(int x, int y) {
    if (x == 0 || y == 0) return 0;
    const int RED_ONE = -1, BLACK_ONE = 1;
    if (x == RED_ONE && y == RED_ONE) return RED_ONE;
    if (x == BLACK_ONE && y == BLACK_ONE) return BLACK_ONE;
    if (x == RED_ONE && y == BLACK_ONE) return 0;
    if (x == BLACK_ONE && y == RED_ONE) return 0;
    if (x == RED_ONE) x = 1;
    if (y == RED_ONE) return (x & 1) ? 0 : x;
    if (y == BLACK_ONE) return (x & 1) ? x : 0;
    int bp = MSB(y);
    return (y & ((1 << bp) - 1)) | (x << bp);
}

// Count how many expanded entries each vertex at the current level will produce.
// count[tid] = |direct_edges(v)| + sum(closure_sizes[w] for non-self successors w)
__global__ void dag_count_expanded(
    const int* __restrict__ level_verts,
    int n_level,
    const int* __restrict__ A_indptr,
    const int* __restrict__ A_indices,
    const int* __restrict__ closure_sizes,
    int* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_level) return;

    int v = level_verts[tid];
    int count = 0;
    for (int e = A_indptr[v]; e < A_indptr[v + 1]; e++) {
        count++;  // direct edge
        int w = A_indices[e];
        if (w != v) count += closure_sizes[w];
    }
    counts[tid] = count;
}

// Expand closure entries for vertices at the current level.
// For each vertex v, writes direct edges + avos_product(edge, closure_entry)
// for all successor closure entries.
__global__ void dag_expand_closures(
    const int* __restrict__ level_verts,
    int n_level,
    const int* __restrict__ A_indptr,
    const int* __restrict__ A_indices,
    const int* __restrict__ A_data,
    const long long* __restrict__ closure_offsets,
    const int* __restrict__ closure_sizes,
    const int* __restrict__ closure_cols,
    const int* __restrict__ closure_vals,
    const long long* __restrict__ output_offsets,
    int* __restrict__ out_rows,
    int* __restrict__ out_cols,
    int* __restrict__ out_vals
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_level) return;

    int v = level_verts[tid];
    long long out_idx = output_offsets[tid];

    for (int e = A_indptr[v]; e < A_indptr[v + 1]; e++) {
        int w = A_indices[e];
        int v_to_w = A_data[e];

        // Direct edge
        out_rows[out_idx] = v;
        out_cols[out_idx] = w;
        out_vals[out_idx] = v_to_w;
        out_idx++;

        if (w == v) continue;  // skip self-loop for propagation

        // Propagate closure[w]
        long long cl_start = closure_offsets[w];
        int cl_size = closure_sizes[w];
        for (int c = 0; c < cl_size; c++) {
            int x_col = closure_cols[cl_start + c];
            int w_to_x = closure_vals[cl_start + c];
            out_rows[out_idx] = v;
            out_cols[out_idx] = x_col;
            out_vals[out_idx] = avos_product(v_to_w, w_to_x);
            out_idx++;
        }
    }
}

// Scatter closure entries from flat storage into CSR-ordered output arrays.
// One thread per vertex — copies closure_cols/vals[src..src+sz] to final[dst..dst+sz].
__global__ void dag_scatter_to_csr(
    const long long* __restrict__ closure_offsets,
    const int* __restrict__ closure_sizes,
    const int* __restrict__ closure_cols,
    const int* __restrict__ closure_vals,
    const int* __restrict__ csr_indptr,
    int* __restrict__ out_indices,
    int* __restrict__ out_data,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int sz = closure_sizes[v];
    if (sz == 0) return;

    long long src = closure_offsets[v];
    int dst = csr_indptr[v];
    for (int i = 0; i < sz; i++) {
        out_indices[dst + i] = closure_cols[src + i];
        out_data[dst + i] = closure_vals[src + i];
    }
}

} // extern "C"
'''

_dag_module = None


def _get_dag_kernels():
    """Compile and cache DAG closure CUDA kernels."""
    global _dag_module
    if _dag_module is None:
        _dag_module = cp.RawModule(code=_DAG_KERNELS_CODE)
    return (
        _dag_module.get_function('dag_count_expanded'),
        _dag_module.get_function('dag_expand_closures'),
        _dag_module.get_function('dag_scatter_to_csr'),
    )


def _compute_levels_cpu(A_cpu):
    """
    Compute topological levels for a DAG.

    Level 0 = sinks (no outgoing non-self edges).
    Level k = vertices whose max successor level is k-1.

    Returns list of int32 arrays, one per level, containing vertex IDs.
    """
    from redblackgraph.sparse.csgraph._topological_sort import topological_sort

    n = A_cpu.shape[0]
    if n == 0:
        return []

    topo = topological_sort(A_cpu)
    indptr = A_cpu.indptr
    indices = A_cpu.indices

    levels = np.zeros(n, dtype=np.int32)

    # Process in reverse topological order (sinks first)
    for i in range(n - 1, -1, -1):
        v = topo[i]
        max_succ = -1
        for idx in range(indptr[v], indptr[v + 1]):
            w = indices[idx]
            if w != v and levels[w] > max_succ:
                max_succ = levels[w]
        levels[v] = max_succ + 1

    max_level = int(levels.max()) if n > 0 else -1
    return [np.where(levels == lev)[0].astype(np.int32) for lev in range(max_level + 1)]


def transitive_closure_dag_gpu(
    A: CSRMatrixGPU,
) -> Tuple[CSRMatrixGPU, int]:
    """
    Compute transitive closure of a DAG using level-parallel propagation on GPU.

    Processes vertices by topological level — all vertices at the same level
    are independent and processed in parallel. At each level, a CUDA kernel
    expands successor closures (applying avos_product), then entries are
    sorted and reduced via AVOS sum.

    Much faster than repeated squaring for DAGs because it does O(V+E+nnz)
    total work instead of O(nnz² · log d).

    Args:
        A: Input adjacency matrix on GPU (must be a DAG)

    Returns:
        Tuple of (closure matrix, diameter estimate)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU operations")

    n = A.shape[0]
    if A.nnz == 0:
        return A.copy(), 0

    # Step 1: Compute topological levels on CPU (fast O(V+E))
    A_cpu = A.to_cpu()
    levels = _compute_levels_cpu(A_cpu)

    if not levels:
        return A.copy(), 0

    count_kernel, expand_kernel, scatter_kernel = _get_dag_kernels()
    block_size = 256

    # Step 2: Initialize closure storage
    closure_offsets = cp.zeros(n, dtype=cp.int64)
    closure_sizes = cp.zeros(n, dtype=cp.int32)
    closure_cols = cp.empty(0, dtype=cp.int32)
    closure_vals = cp.empty(0, dtype=cp.int32)
    total_closure = 0

    # Step 3: Process each level (sinks first)
    for level_verts_cpu in levels:
        n_level = len(level_verts_cpu)
        if n_level == 0:
            continue

        level_verts = cp.asarray(level_verts_cpu, dtype=cp.int32)
        grid_size = (n_level + block_size - 1) // block_size

        # 3a: Count expanded entries per vertex (upper bound)
        counts = cp.empty(n_level, dtype=cp.int32)
        count_kernel(
            (grid_size,), (block_size,),
            (level_verts, n_level, A.indptr, A.indices, closure_sizes, counts)
        )

        # 3b: Prefix sum for output offsets
        output_offsets = cp.zeros(n_level + 1, dtype=cp.int64)
        output_offsets[1:] = cp.cumsum(counts.astype(cp.int64))
        total_expanded = int(output_offsets[-1].get())

        if total_expanded == 0:
            continue

        # 3c: Expand — direct edges + avos_product(edge, successor_closure)
        out_rows = cp.empty(total_expanded, dtype=cp.int32)
        out_cols = cp.empty(total_expanded, dtype=cp.int32)
        out_vals = cp.empty(total_expanded, dtype=cp.int32)

        expand_kernel(
            (grid_size,), (block_size,),
            (level_verts, n_level,
             A.indptr, A.indices, A.data,
             closure_offsets, closure_sizes, closure_cols, closure_vals,
             output_offsets, out_rows, out_cols, out_vals)
        )

        # 3d: Filter zeros (from avos_product returning 0)
        nonzero = out_vals != 0
        if not cp.all(nonzero):
            out_rows = out_rows[nonzero]
            out_cols = out_cols[nonzero]
            out_vals = out_vals[nonzero]
            total_expanded = len(out_vals)
            if total_expanded == 0:
                continue

        # 3e: Sort by (vertex, column) for deduplication
        sort_key = out_rows.astype(cp.int64) * n + out_cols.astype(cp.int64)
        order = cp.argsort(sort_key)
        out_rows = out_rows[order]
        out_cols = out_cols[order]
        out_vals = out_vals[order]
        sort_key = sort_key[order]

        # 3f: Find unique (vertex, col) positions and reduce with AVOS sum
        key_changes = cp.concatenate([cp.array([True]), sort_key[1:] != sort_key[:-1]])
        unique_idx = cp.where(key_changes)[0]

        reduced_vals = _reduce_avos_sum(out_vals, unique_idx, total_expanded)
        reduced_rows = out_rows[unique_idx]
        reduced_cols = out_cols[unique_idx]

        # Filter any remaining zeros after reduction
        nz_mask = reduced_vals != 0
        if not cp.all(nz_mask):
            reduced_rows = reduced_rows[nz_mask]
            reduced_cols = reduced_cols[nz_mask]
            reduced_vals = reduced_vals[nz_mask]

        n_reduced = len(reduced_cols)
        if n_reduced == 0:
            continue

        # 3g: Update closure storage for this level's vertices
        # Entries are sorted by (row, col), so same-vertex entries are contiguous
        vert_changes = cp.concatenate([
            cp.array([True]),
            reduced_rows[1:] != reduced_rows[:-1]
        ])
        vert_starts = cp.where(vert_changes)[0]
        vert_ids = reduced_rows[vert_starts]
        vert_ends = cp.concatenate([
            vert_starts[1:],
            cp.array([n_reduced], dtype=cp.int64)
        ])
        vert_counts = (vert_ends - vert_starts).astype(cp.int32)

        closure_offsets[vert_ids] = vert_starts.astype(cp.int64) + total_closure
        closure_sizes[vert_ids] = vert_counts

        # Append to flat arrays
        closure_cols = cp.concatenate([closure_cols, reduced_cols.astype(cp.int32)])
        closure_vals = cp.concatenate([closure_vals, reduced_vals.astype(cp.int32)])
        total_closure += n_reduced

    # Step 4: Build final CSR matrix from flat closure arrays
    if total_closure == 0:
        empty_indptr = cp.zeros(n + 1, dtype=cp.int32)
        return CSRMatrixGPU(
            cp.empty(0, dtype=cp.int32),
            cp.empty(0, dtype=cp.int32),
            empty_indptr, (n, n),
            triangular=A.triangular, validate=False
        ), 0

    # Gather all closure entries per vertex into CSR order via GPU scatter kernel
    final_indptr = cp.zeros(n + 1, dtype=cp.int32)
    final_indptr[1:] = cp.cumsum(closure_sizes)
    total_nnz = int(final_indptr[-1].get())

    final_indices = cp.empty(total_nnz, dtype=cp.int32)
    final_data = cp.empty(total_nnz, dtype=cp.int32)

    grid_size = (n + block_size - 1) // block_size
    scatter_kernel(
        (grid_size,), (block_size,),
        (closure_offsets, closure_sizes, closure_cols, closure_vals,
         final_indptr, final_indices, final_data, n)
    )

    # Estimate diameter from max value
    diameter = 0
    if total_nnz > 0:
        abs_vals = cp.abs(final_data)
        non_identity = abs_vals[abs_vals > 1]
        if len(non_identity) > 0:
            max_val = int(cp.max(non_identity).get())
            diameter = int(np.floor(np.log2(max_val))) if max_val > 1 else 0

    result = CSRMatrixGPU(
        final_data, final_indices, final_indptr, (n, n),
        triangular=A.triangular, validate=False
    )
    return result, diameter
