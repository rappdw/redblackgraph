"""
GPU-resident transitive closure via repeated squaring.

Computes TC(A) = A + A² + A⁴ + A⁸ + ... using AVOS semiring operations,
keeping all data on GPU between iterations (no CPU round-trips).

Algorithm mirrors sparse/csgraph/transitive_closure.py:transitive_closure_squaring()
but uses CSRMatrixGPU and GPU SpGEMM throughout.
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
