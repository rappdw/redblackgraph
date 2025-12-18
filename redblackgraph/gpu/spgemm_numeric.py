"""
Numeric phase of SpGEMM (Sparse General Matrix-Matrix Multiplication).

This module implements the value computation phase for C = A @ A where:
- Pattern is known from symbolic phase (hash tables with column indices)
- Computes actual AVOS product values using global memory hash tables
- Uses atomicMin for deterministic AVOS sum reduction

The numeric phase computes:
1. Column indices for each row (extracted from hash tables)
2. Values using AVOS sum and product operations
3. Sorted output via global sort for determinism
"""

import numpy as np
from typing import Tuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# Sentinel value for AVOS "zero" in hash tables
# AVOS sum is "non-zero minimum", so we use INT_MAX to represent zero
# This allows us to use atomicMin for reduction
AVOS_ZERO_SENTINEL = 2147483647  # INT_MAX


# CUDA kernel for numeric phase with AVOS operations using global memory hash tables
NUMERIC_HASH_KERNEL = r'''
extern "C" {

// Sentinel value for AVOS "zero" (must match Python constant)
#define AVOS_ZERO_SENTINEL 2147483647

// Simple hash function (must match symbolic phase)
__device__ inline unsigned int hash_func(int key, int table_size) {
    unsigned int h = (unsigned int)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & (table_size - 1);  // table_size must be power of 2
}

// Helper function: Find MSB position
__device__ inline int MSB(int x) {
    int bit_position = 0;
    while (x > 1) {
        x >>= 1;
        bit_position++;
    }
    return bit_position;
}

// AVOS product with parity constraints
__device__ inline int avos_product(int x, int y) {
    if (x == 0 || y == 0) return 0;

    const int RED_ONE = -1;
    const int BLACK_ONE = 1;

    // Identity ⊗ Identity special cases
    if (x == RED_ONE && y == RED_ONE) return RED_ONE;
    if (x == BLACK_ONE && y == BLACK_ONE) return BLACK_ONE;
    if (x == RED_ONE && y == BLACK_ONE) return 0;
    if (x == BLACK_ONE && y == RED_ONE) return 0;

    // LEFT identity: treat as 1 for composition
    if (x == RED_ONE) x = 1;

    // RIGHT identity: parity filter
    if (y == RED_ONE) {
        return (x & 1) ? 0 : x;
    }
    if (y == BLACK_ONE) {
        return (x & 1) ? x : 0;
    }

    // General case: bit shifting composition
    int bit_position = MSB(y);
    int mask = (1 << bit_position) - 1;
    return (y & mask) | (x << bit_position);
}

// Transform value for atomicMin: map 0 to INT_MAX, keep others as-is
// This allows atomicMin to compute AVOS sum (non-zero minimum)
__device__ inline int to_min_space(int val) {
    return (val == 0) ? AVOS_ZERO_SENTINEL : val;
}

// Inverse transform: map INT_MAX back to 0
__device__ inline int from_min_space(int val) {
    return (val == AVOS_ZERO_SENTINEL) ? 0 : val;
}

// Numeric phase kernel using global memory hash tables
// Accumulates AVOS products into hash tables using atomicMin
__global__ void numeric_hash_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    const int* __restrict__ dataA,
    const int* __restrict__ hash_keys,      // From symbolic phase (column indices)
    int* __restrict__ hash_vals,            // Values to accumulate (initialized to AVOS_ZERO_SENTINEL)
    const long long* __restrict__ table_offsets,
    const int* __restrict__ table_sizes,
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

    // For each non-zero A[i,k] in row i
    for (int k_idx = row_start; k_idx < row_end; k_idx++) {
        int k = indicesA[k_idx];
        int val_ik = dataA[k_idx];

        // Get row k of A
        int k_row_start = indptrA[k];
        int k_row_end = indptrA[k + 1];

        // For each non-zero A[k,j] in row k
        for (int j_idx = k_row_start; j_idx < k_row_end; j_idx++) {
            int j = indicesA[j_idx];
            int val_kj = dataA[j_idx];

            // Apply triangular mask: only j >= row_i
            if (j >= row_i) {
                // Compute A[i,k] ⊗ A[k,j]
                int prod = avos_product(val_ik, val_kj);

                if (prod != 0) {
                    // Find column j in hash table (it must exist from symbolic phase)
                    unsigned int h = hash_func(j, size);

                    for (int probe = 0; probe < size; probe++) {
                        long long idx = base + ((h + probe) & (size - 1));

                        if (hash_keys[idx] == j) {
                            // Found it - accumulate using atomicMin
                            // AVOS sum is non-zero minimum, so we use atomicMin
                            // with transformed values (0 -> INT_MAX)
                            int prod_t = to_min_space(prod);
                            atomicMin(&hash_vals[idx], prod_t);
                            break;
                        }

                        if (hash_keys[idx] == -1) {
                            // Empty slot - column j not in pattern (shouldn't happen)
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Extract entries from hash tables into output arrays
__global__ void extract_hash_entries_kernel(
    const int* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    const long long* __restrict__ table_offsets,
    const int* __restrict__ table_sizes,
    const int* __restrict__ indptrC,
    int* __restrict__ out_rows,
    int* __restrict__ out_cols,
    int* __restrict__ out_vals,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_i >= n_rows) return;

    long long base = table_offsets[row_i];
    int size = table_sizes[row_i];
    int out_start = indptrC[row_i];

    int out_idx = out_start;
    for (int i = 0; i < size; i++) {
        int key = hash_keys[base + i];
        if (key != -1) {
            int val = hash_vals[base + i];
            // Transform back from min space
            val = (val == AVOS_ZERO_SENTINEL) ? 0 : val;

            out_rows[out_idx] = row_i;
            out_cols[out_idx] = key;
            out_vals[out_idx] = val;
            out_idx++;
        }
    }
}

} // extern "C"
'''


class NumericPhase:
    """
    Numeric phase implementation for SpGEMM using global memory hash tables.

    Computes actual values for C = A @ A using AVOS operations.
    No arbitrary limit on unique output columns per row.
    """

    def __init__(self):
        """Initialize and compile numeric phase kernels."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")

        self._module = cp.RawModule(code=NUMERIC_HASH_KERNEL)
        self._hash_kernel = self._module.get_function('numeric_hash_kernel')
        self._extract_kernel = self._module.get_function('extract_hash_entries_kernel')

    def compute_values(
        self,
        indptrA: 'cp.ndarray',
        indicesA: 'cp.ndarray',
        dataA: 'cp.ndarray',
        indptrC: 'cp.ndarray',
        nnzC: int,
        n_rows: int,
        hash_keys: 'cp.ndarray',
        table_offsets: 'cp.ndarray',
        table_sizes: 'cp.ndarray'
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """
        Compute column indices and values for C = A @ A using hash tables.

        Args:
            indptrA: Row pointers of A
            indicesA: Column indices of A
            dataA: Values of A (int32)
            indptrC: Row pointers of C (from symbolic phase)
            nnzC: Total non-zeros in C (from symbolic phase)
            n_rows: Number of rows
            hash_keys: Hash table keys from symbolic phase
            table_offsets: Start offset for each row's hash table
            table_sizes: Size of each row's hash table

        Returns:
            indicesC: Column indices of C (int32, sorted within each row)
            dataC: Values of C (int32)
        """
        if nnzC == 0:
            return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)

        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size

        # Convert to appropriate types
        indptrA_i32 = cp.asarray(indptrA, dtype=cp.int32)
        indicesA_i32 = cp.asarray(indicesA, dtype=cp.int32)
        dataA_i32 = cp.asarray(dataA, dtype=cp.int32)
        indptrC_i32 = cp.asarray(indptrC, dtype=cp.int32)

        # Allocate hash values array (initialized to AVOS_ZERO_SENTINEL)
        total_hash_size = len(hash_keys)
        if total_hash_size > 0:
            hash_vals = cp.full(total_hash_size, AVOS_ZERO_SENTINEL, dtype=cp.int32)
        else:
            hash_vals = cp.array([], dtype=cp.int32)

        # Step 1: Accumulate values into hash tables
        if total_hash_size > 0:
            self._hash_kernel(
                (grid_size,), (block_size,),
                (indptrA_i32, indicesA_i32, dataA_i32,
                 hash_keys, hash_vals, table_offsets, table_sizes, n_rows)
            )

        # Step 2: Extract entries from hash tables
        out_rows = cp.zeros(nnzC, dtype=cp.int32)
        out_cols = cp.zeros(nnzC, dtype=cp.int32)
        out_vals = cp.zeros(nnzC, dtype=cp.int32)

        if total_hash_size > 0:
            self._extract_kernel(
                (grid_size,), (block_size,),
                (hash_keys, hash_vals, table_offsets, table_sizes,
                 indptrC_i32, out_rows, out_cols, out_vals, n_rows)
            )

        # Step 3: Sort by (row, col) for deterministic output
        # Create composite sort key: row * n_cols + col
        n_cols = n_rows  # Square matrix
        sort_key = out_rows.astype(cp.int64) * n_cols + out_cols.astype(cp.int64)
        order = cp.argsort(sort_key)

        indicesC = out_cols[order].astype(cp.int32)
        dataC = out_vals[order].astype(cp.int32)

        return indicesC, dataC


def compute_numeric_values(
    indptrA: 'cp.ndarray',
    indicesA: 'cp.ndarray',
    dataA: 'cp.ndarray',
    indptrC: 'cp.ndarray',
    nnzC: int,
    n_rows: int,
    hash_keys: 'cp.ndarray',
    table_offsets: 'cp.ndarray',
    table_sizes: 'cp.ndarray'
) -> Tuple['cp.ndarray', 'cp.ndarray']:
    """
    Compute numeric values for C = A @ A (upper triangular).

    This is the main entry point for the numeric phase.

    Args:
        indptrA: Row pointers of A
        indicesA: Column indices of A
        dataA: Values of A
        indptrC: Row pointers of C (from symbolic phase)
        nnzC: Total non-zeros in C (from symbolic phase)
        n_rows: Number of rows
        hash_keys: Hash table keys from symbolic phase
        table_offsets: Start offset for each row's hash table
        table_sizes: Size of each row's hash table

    Returns:
        indicesC: Column indices of C (sorted within each row)
        dataC: Values of C
    """
    numeric = NumericPhase()
    return numeric.compute_values(
        indptrA, indicesA, dataA,
        indptrC, nnzC, n_rows,
        hash_keys, table_offsets, table_sizes
    )


# Singleton instance
_numeric = None


def get_numeric_phase() -> NumericPhase:
    """Get or create singleton numeric phase instance."""
    global _numeric
    if _numeric is None:
        _numeric = NumericPhase()
    return _numeric
