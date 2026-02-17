"""
Symbolic phase of SpGEMM (Sparse General Matrix-Matrix Multiplication).

This module implements the pattern computation phase for C = A @ A where:
- A is upper triangular CSR matrix
- Result C is also upper triangular
- Uses global memory hash tables for unlimited output columns per row

The symbolic phase computes:
1. How many non-zeros each row of C will have
2. Which columns will be non-zero (the pattern)

This allows us to allocate C's CSR arrays before the numeric phase.
"""

import numpy as np
from typing import Tuple

from ._cuda_utils import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
else:
    cp = None


# CUDA kernel to count candidates per row (upper bound on unique outputs)
COUNT_CANDIDATES_KERNEL = r'''
extern "C" {

// Count the number of candidate contributions for each row
// candidates[i] = sum over k in row i of A of nnz(row k of A)
// This is an upper bound on unique output columns per row
__global__ void count_candidates_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    int* __restrict__ candidates,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_i >= n_rows) return;

    int row_start = indptrA[row_i];
    int row_end = indptrA[row_i + 1];

    int count = 0;
    for (int k_idx = row_start; k_idx < row_end; k_idx++) {
        int k = indicesA[k_idx];
        // Count entries in row k of A that satisfy triangular mask (j >= row_i)
        int k_row_start = indptrA[k];
        int k_row_end = indptrA[k + 1];

        // For upper triangular, we need j >= row_i
        // Since A is upper triangular, all j in row k satisfy j >= k
        // We need j >= row_i, and since k >= row_i (from upper triangular A[i,k]),
        // and j >= k, we have j >= k >= row_i, so all entries qualify
        count += (k_row_end - k_row_start);
    }

    candidates[row_i] = count;
}

} // extern "C"
'''


# CUDA kernel for symbolic phase using global memory hash tables
SYMBOLIC_HASH_KERNEL = r'''
extern "C" {

// Simple hash function
__device__ inline unsigned int hash_func(int key, int table_size) {
    unsigned int h = (unsigned int)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & (table_size - 1);  // table_size must be power of 2
}

// Symbolic phase kernel using global memory hash tables
// Inserts unique column indices into per-row hash tables
__global__ void symbolic_hash_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    int* __restrict__ hash_keys,      // Global hash table keys (initialized to -1)
    const long long* __restrict__ table_offsets,  // Start offset for each row's hash table
    const int* __restrict__ table_sizes,    // Size of each row's hash table (power of 2)
    int* __restrict__ overflow_flag,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_i >= n_rows) return;

    int row_start = indptrA[row_i];
    int row_end = indptrA[row_i + 1];

    if (row_start == row_end) return;  // Empty row

    long long base = table_offsets[row_i];
    int size = table_sizes[row_i];

    if (size == 0) return;  // No hash table allocated

    // For each non-zero in row i of A
    for (int k_idx = row_start; k_idx < row_end; k_idx++) {
        int k = indicesA[k_idx];

        // Add columns from row k of A
        int k_row_start = indptrA[k];
        int k_row_end = indptrA[k + 1];

        for (int j_idx = k_row_start; j_idx < k_row_end; j_idx++) {
            int j = indicesA[j_idx];

            // Apply triangular mask: only j >= row_i
            if (j >= row_i) {
                // Insert j into hash table using linear probing
                unsigned int h = hash_func(j, size);

                for (int probe = 0; probe < size; probe++) {
                    long long idx = base + ((h + probe) & (size - 1));

                    int old = atomicCAS(&hash_keys[idx], -1, j);
                    if (old == -1 || old == j) {
                        // Successfully inserted or already exists
                        break;
                    }

                    if (probe == size - 1) {
                        // Hash table full - shouldn't happen with proper sizing
                        atomicExch(overflow_flag, row_i + 1);
                    }
                }
            }
        }
    }
}

// Count non-empty entries in each row's hash table
__global__ void count_hash_entries_kernel(
    const int* __restrict__ hash_keys,
    const long long* __restrict__ table_offsets,
    const int* __restrict__ table_sizes,
    int* __restrict__ row_nnz,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_i >= n_rows) return;

    long long base = table_offsets[row_i];
    int size = table_sizes[row_i];

    int count = 0;
    for (int i = 0; i < size; i++) {
        if (hash_keys[base + i] != -1) {
            count++;
        }
    }

    row_nnz[row_i] = count;
}

} // extern "C"
'''


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class SymbolicPhase:
    """
    Symbolic phase implementation for SpGEMM using global memory hash tables.

    Computes the sparsity pattern for C = A @ A (upper triangular).
    No arbitrary limit on unique output columns per row.
    """

    def __init__(self):
        """Initialize and compile symbolic phase kernels."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")

        self._count_module = cp.RawModule(code=COUNT_CANDIDATES_KERNEL)
        self._count_kernel = self._count_module.get_function('count_candidates_kernel')

        self._hash_module = cp.RawModule(code=SYMBOLIC_HASH_KERNEL)
        self._hash_kernel = self._hash_module.get_function('symbolic_hash_kernel')
        self._count_entries_kernel = self._hash_module.get_function('count_hash_entries_kernel')

    def compute_pattern(
        self,
        indptrA: 'cp.ndarray',
        indicesA: 'cp.ndarray',
        n_rows: int
    ) -> Tuple['cp.ndarray', 'cp.ndarray', 'cp.ndarray', 'cp.ndarray']:
        """
        Compute row non-zero counts for C = A @ A using global memory hash tables.

        Args:
            indptrA: Row pointers of A (int32 or int64)
            indicesA: Column indices of A (int32)
            n_rows: Number of rows in A

        Returns:
            row_nnz: Array of length n_rows with count of non-zeros per row
            hash_keys: Global hash table keys (for reuse in numeric phase)
            table_offsets: Start offset for each row's hash table
            table_sizes: Size of each row's hash table

        Raises:
            RuntimeError: If hash table overflow occurs (shouldn't happen with proper sizing)
        """
        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size

        # Convert to int32 if needed
        indptrA_i32 = cp.asarray(indptrA, dtype=cp.int32)
        indicesA_i32 = cp.asarray(indicesA, dtype=cp.int32)

        # Step 1: Count candidates per row (upper bound on unique outputs)
        candidates = cp.zeros(n_rows, dtype=cp.int32)
        self._count_kernel(
            (grid_size,), (block_size,),
            (indptrA_i32, indicesA_i32, candidates, n_rows)
        )

        # Step 2: Compute hash table sizes (next power of 2, with load factor 0.5)
        # table_size = next_pow2(max(1, 2 * candidates))
        candidates_cpu = candidates.get()
        table_sizes_cpu = np.array([
            _next_power_of_2(max(1, 2 * c)) if c > 0 else 0
            for c in candidates_cpu
        ], dtype=np.int32)

        # Step 3: Compute table offsets (prefix sum) - use int64 for large tables
        table_offsets_cpu = np.zeros(n_rows + 1, dtype=np.int64)
        table_offsets_cpu[1:] = np.cumsum(table_sizes_cpu.astype(np.int64))
        total_hash_size = int(table_offsets_cpu[-1])

        # Check memory usage and warn if very large
        hash_memory_mb = total_hash_size * 4 / (1024 * 1024)
        if hash_memory_mb > 1000:  # > 1GB
            import warnings
            warnings.warn(
                f"SpGEMM symbolic phase allocating {hash_memory_mb:.1f} MB for hash tables. "
                f"This may cause out-of-memory errors on some GPUs."
            )

        table_sizes = cp.asarray(table_sizes_cpu, dtype=cp.int32)
        table_offsets = cp.asarray(table_offsets_cpu[:-1], dtype=cp.int64)  # Don't need last element

        # Step 4: Allocate and initialize hash tables
        if total_hash_size > 0:
            hash_keys = cp.full(total_hash_size, -1, dtype=cp.int32)
        else:
            hash_keys = cp.array([], dtype=cp.int32)

        overflow_flag = cp.zeros(1, dtype=cp.int32)

        # Step 5: Insert column indices into hash tables
        if total_hash_size > 0:
            self._hash_kernel(
                (grid_size,), (block_size,),
                (indptrA_i32, indicesA_i32, hash_keys, table_offsets, table_sizes, overflow_flag, n_rows)
            )

            # Check for overflow
            overflow_row = int(overflow_flag[0].get())
            if overflow_row > 0:
                raise RuntimeError(
                    f"SpGEMM symbolic phase hash table overflow at row {overflow_row - 1}. "
                    f"This indicates a bug in hash table sizing."
                )

        # Step 6: Count entries in each hash table
        row_nnz = cp.zeros(n_rows, dtype=cp.int32)
        if total_hash_size > 0:
            self._count_entries_kernel(
                (grid_size,), (block_size,),
                (hash_keys, table_offsets, table_sizes, row_nnz, n_rows)
            )

        return row_nnz, hash_keys, table_offsets, table_sizes


def prefix_sum_scan(row_nnz: 'cp.ndarray') -> 'cp.ndarray':
    """
    Compute exclusive prefix sum (scan) to build indptr from row counts.

    Given row_nnz = [3, 0, 2, 1], returns indptr = [0, 3, 3, 5, 6]

    Args:
        row_nnz: Number of non-zeros per row (int32)

    Returns:
        indptr: Row pointers for CSR (int32), length = len(row_nnz) + 1
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required")

    n_rows = len(row_nnz)

    # Use CuPy's cumsum for the scan
    # Exclusive scan: prepend 0 and don't include the last cumsum value
    indptr = cp.empty(n_rows + 1, dtype=cp.int32)
    indptr[0] = 0
    indptr[1:] = cp.cumsum(row_nnz, dtype=cp.int32)

    return indptr


def compute_symbolic_pattern(
    indptrA: 'cp.ndarray',
    indicesA: 'cp.ndarray',
    n_rows: int,
    n_cols: int
) -> Tuple['cp.ndarray', int]:
    """
    Compute symbolic pattern for C = A @ A (upper triangular).

    This is the main entry point for the symbolic phase.

    Args:
        indptrA: Row pointers of A
        indicesA: Column indices of A
        n_rows: Number of rows in A
        n_cols: Number of columns in A

    Returns:
        indptrC: Row pointers for C (length n_rows + 1)
        nnzC: Total number of non-zeros in C
    """
    indptrC, nnzC, _, _, _ = compute_symbolic_pattern_with_tables(
        indptrA, indicesA, n_rows, n_cols
    )
    return indptrC, nnzC


def compute_symbolic_pattern_with_tables(
    indptrA: 'cp.ndarray',
    indicesA: 'cp.ndarray',
    n_rows: int,
    n_cols: int
) -> Tuple['cp.ndarray', int, 'cp.ndarray', 'cp.ndarray', 'cp.ndarray']:
    """
    Compute symbolic pattern for C = A @ A (upper triangular), returning
    additional data structures needed by the numeric phase.

    Returns:
        indptrC: Row pointers for C (length n_rows + 1)
        nnzC: Total number of non-zeros in C
        hash_keys: Global hash table keys (for reuse in numeric phase)
        table_offsets: Start offset for each row's hash table
        table_sizes: Size of each row's hash table
    """
    # Compute per-row non-zero counts using hash tables
    symbolic = SymbolicPhase()
    row_nnz, hash_keys, table_offsets, table_sizes = symbolic.compute_pattern(
        indptrA, indicesA, n_rows
    )

    # Build indptr via prefix sum
    indptrC = prefix_sum_scan(row_nnz)

    # Total nnz is the last element of indptr
    nnzC = int(indptrC[-1].get())

    return indptrC, nnzC, hash_keys, table_offsets, table_sizes


# Singleton instance
_symbolic = None


def get_symbolic_phase() -> SymbolicPhase:
    """Get or create singleton symbolic phase instance."""
    global _symbolic
    if _symbolic is None:
        _symbolic = SymbolicPhase()
    return _symbolic
