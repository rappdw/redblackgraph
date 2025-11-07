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


# CUDA kernel for numeric phase with AVOS operations
NUMERIC_KERNEL = r'''
extern "C" {

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

// Find intersection of two sorted lists and compute AVOS sum-product
// Returns accumulated value using AVOS operations
__device__ int compute_inner_product(
    const int* __restrict__ indicesA,
    const int* __restrict__ dataA,
    int row_i_start,
    int row_i_end,
    int row_j_start,
    int row_j_end
) {
    int acc = 0;  // AVOS additive identity
    int i = row_i_start;
    int j = row_j_start;
    
    // Merge the two sorted index lists to find common k values
    while (i < row_i_end && j < row_j_end) {
        int k_i = indicesA[i];
        int k_j = indicesA[j];
        
        if (k_i < k_j) {
            i++;
        } else if (k_i > k_j) {
            j++;
        } else {
            // Common index k: compute A[row_i, k] ⊗ A[k, row_j]
            int val_i = dataA[i];
            int val_j = dataA[j];
            int prod = avos_product(val_i, val_j);
            acc = avos_sum(acc, prod);
            i++;
            j++;
        }
    }
    
    return acc;
}

// Numeric phase kernel - one row per thread
// Fills indicesC and dataC based on pattern from symbolic phase
__global__ void numeric_phase_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    const int* __restrict__ dataA,
    const int* __restrict__ indptrC,
    int* __restrict__ indicesC,
    int* __restrict__ dataC,
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
    
    // For this row, collect all candidate columns and compute values
    // Using a simple bitmap approach (limited to 1024 cols for prototype)
    const int MAX_COLS = 1024;
    int col_values[MAX_COLS];
    bool col_present[MAX_COLS];
    
    // Initialize
    for (int c = 0; c < MAX_COLS; c++) {
        col_values[c] = 0;
        col_present[c] = false;
    }
    
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
            if (j >= row_i && j < MAX_COLS) {
                // Compute A[i,k] ⊗ A[k,j]
                int prod = avos_product(val_ik, val_kj);
                
                // Accumulate: C[i,j] += prod (using AVOS sum)
                if (col_present[j]) {
                    col_values[j] = avos_sum(col_values[j], prod);
                } else {
                    col_values[j] = prod;
                    col_present[j] = true;
                }
            }
        }
    }
    
    // Write results to output in sorted column order
    int out_idx = out_start;
    for (int j = row_i; j < MAX_COLS && out_idx < out_end; j++) {
        if (col_present[j] && col_values[j] != 0) {
            indicesC[out_idx] = j;
            dataC[out_idx] = col_values[j];
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
        """
        # Allocate output arrays
        indicesC = cp.zeros(nnzC, dtype=cp.int32)
        dataC = cp.zeros(nnzC, dtype=cp.int32)
        
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
             indptrC_i32, indicesC, dataC, n_rows)
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
