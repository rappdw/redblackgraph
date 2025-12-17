"""
Numeric phase of SpGEMM (Sparse General Matrix-Matrix Multiplication).

This module implements the value computation phase for C = A @ A where:
- Pattern is known from symbolic phase (indptrC)
- Computes actual AVOS product values
- Uses deterministic merge-based approach

The numeric phase computes:
1. Column indices for each row (in sorted order)
2. Values using AVOS sum and product operations
"""

import numpy as np
from typing import Tuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# Maximum unique output columns per row (must match symbolic phase)
# This limits memory per thread, not the column index range
MAX_UNIQUE_PER_ROW = 512


# CUDA kernel for numeric phase with AVOS operations
NUMERIC_KERNEL = r'''
extern "C" {

// Maximum unique output columns per row (must match Python constant)
#define MAX_UNIQUE_PER_ROW 512

// AVOS sum: Non-zero minimum
__device__ inline int avos_sum(int x, int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    return (x < y) ? x : y;
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

// Simple insertion sort for small arrays (sorts by column index)
__device__ void insertion_sort_by_col(int* cols, int* vals, int n) {
    for (int i = 1; i < n; i++) {
        int key_col = cols[i];
        int key_val = vals[i];
        int j = i - 1;
        
        while (j >= 0 && cols[j] > key_col) {
            cols[j + 1] = cols[j];
            vals[j + 1] = vals[j];
            j--;
        }
        cols[j + 1] = key_col;
        vals[j + 1] = key_val;
    }
}

// Numeric phase kernel - one row per thread
// Fills indicesC and dataC based on pattern from symbolic phase
// Uses dynamic per-row accumulator keyed by actual column indices
__global__ void numeric_phase_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    const int* __restrict__ dataA,
    const int* __restrict__ indptrC,
    int* __restrict__ indicesC,
    int* __restrict__ dataC,
    int* __restrict__ overflow_flag,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_i >= n_rows) return;
    
    // Get range for this row in A
    int row_i_start = indptrA[row_i];
    int row_i_end = indptrA[row_i + 1];
    
    if (row_i_start == row_i_end) {
        return;  // Empty row in A means empty row in C
    }
    
    // Get range for this row in C (output)
    int out_start = indptrC[row_i];
    int out_end = indptrC[row_i + 1];
    
    if (out_start == out_end) {
        return;  // No output for this row
    }
    
    // Use dynamic arrays to track (column, value) pairs
    // This is keyed by actual column indices, not limited by column index range
    int acc_cols[MAX_UNIQUE_PER_ROW];
    int acc_vals[MAX_UNIQUE_PER_ROW];
    int num_entries = 0;
    
    // For each non-zero A[i,k] in row i
    for (int k_idx = row_i_start; k_idx < row_i_end; k_idx++) {
        int k = indicesA[k_idx];
        int val_ik = dataA[k_idx];
        
        // Get row k of A
        int row_k_start = indptrA[k];
        int row_k_end = indptrA[k + 1];
        
        // For each non-zero A[k,j] in row k
        for (int j_idx = row_k_start; j_idx < row_k_end; j_idx++) {
            int j = indicesA[j_idx];
            int val_kj = dataA[j_idx];
            
            // Apply triangular mask: only j >= row_i
            if (j >= row_i) {
                // Compute A[i,k] ⊗ A[k,j]
                int prod = avos_product(val_ik, val_kj);
                
                if (prod != 0) {
                    // Check if column j is already in our accumulator (linear search)
                    int found_idx = -1;
                    for (int u = 0; u < num_entries; u++) {
                        if (acc_cols[u] == j) {
                            found_idx = u;
                            break;
                        }
                    }
                    
                    if (found_idx >= 0) {
                        // Accumulate: C[i,j] += prod (using AVOS sum)
                        acc_vals[found_idx] = avos_sum(acc_vals[found_idx], prod);
                    } else {
                        // Add new entry
                        if (num_entries < MAX_UNIQUE_PER_ROW) {
                            acc_cols[num_entries] = j;
                            acc_vals[num_entries] = prod;
                            num_entries++;
                        } else {
                            // Overflow - set flag
                            atomicExch(overflow_flag, row_i + 1);
                        }
                    }
                }
            }
        }
    }
    
    // Sort by column index to maintain CSR invariant
    insertion_sort_by_col(acc_cols, acc_vals, num_entries);
    
    // Write results to output (already sorted by column)
    int out_idx = out_start;
    for (int u = 0; u < num_entries && out_idx < out_end; u++) {
        if (acc_vals[u] != 0) {
            indicesC[out_idx] = acc_cols[u];
            dataC[out_idx] = acc_vals[u];
            out_idx++;
        }
    }
}

} // extern "C"
'''


class NumericPhase:
    """
    Numeric phase implementation for SpGEMM.
    
    Computes actual values for C = A @ A using AVOS operations.
    """
    
    def __init__(self):
        """Initialize and compile numeric phase kernel."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")
        
        self._module = cp.RawModule(code=NUMERIC_KERNEL)
        self._kernel = self._module.get_function('numeric_phase_kernel')
    
    def compute_values(
        self,
        indptrA: 'cp.ndarray',
        indicesA: 'cp.ndarray',
        dataA: 'cp.ndarray',
        indptrC: 'cp.ndarray',
        nnzC: int,
        n_rows: int
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """
        Compute column indices and values for C = A @ A.
        
        Args:
            indptrA: Row pointers of A
            indicesA: Column indices of A
            dataA: Values of A (int32)
            indptrC: Row pointers of C (from symbolic phase)
            nnzC: Total non-zeros in C (from symbolic phase)
            n_rows: Number of rows
        
        Returns:
            indicesC: Column indices of C (int32)
            dataC: Values of C (int32)
        
        Raises:
            RuntimeError: If any row exceeds MAX_UNIQUE_PER_ROW unique output columns
        """
        # Allocate output arrays
        indicesC = cp.zeros(nnzC, dtype=cp.int32)
        dataC = cp.zeros(nnzC, dtype=cp.int32)
        
        # Allocate overflow flag (0 = no overflow, row+1 = overflow at row)
        overflow_flag = cp.zeros(1, dtype=cp.int32)
        
        # Launch kernel - one thread per row
        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size
        
        # Convert to int32 if needed
        indptrA_i32 = cp.asarray(indptrA, dtype=cp.int32)
        indicesA_i32 = cp.asarray(indicesA, dtype=cp.int32)
        dataA_i32 = cp.asarray(dataA, dtype=cp.int32)
        indptrC_i32 = cp.asarray(indptrC, dtype=cp.int32)
        
        self._kernel(
            (grid_size,), (block_size,),
            (indptrA_i32, indicesA_i32, dataA_i32,
             indptrC_i32, indicesC, dataC, overflow_flag, n_rows)
        )
        
        # Check for overflow
        overflow_row = int(overflow_flag[0].get())
        if overflow_row > 0:
            raise RuntimeError(
                f"SpGEMM numeric phase overflow: row {overflow_row - 1} exceeded "
                f"maximum of {MAX_UNIQUE_PER_ROW} unique output columns. "
                f"This typically indicates a very dense graph that exceeds "
                f"the expected genealogy workload characteristics."
            )
        
        return indicesC, dataC


def compute_numeric_values(
    indptrA: 'cp.ndarray',
    indicesA: 'cp.ndarray',
    dataA: 'cp.ndarray',
    indptrC: 'cp.ndarray',
    nnzC: int,
    n_rows: int
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
    
    Returns:
        indicesC: Column indices of C
        dataC: Values of C
    """
    numeric = NumericPhase()
    return numeric.compute_values(
        indptrA, indicesA, dataA,
        indptrC, nnzC, n_rows
    )


# Singleton instance
_numeric = None


def get_numeric_phase() -> NumericPhase:
    """Get or create singleton numeric phase instance."""
    global _numeric
    if _numeric is None:
        _numeric = NumericPhase()
    return _numeric
