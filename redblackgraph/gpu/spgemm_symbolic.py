"""
Symbolic phase of SpGEMM (Sparse General Matrix-Matrix Multiplication).

This module implements the pattern computation phase for C = A @ A where:
- A is upper triangular CSR matrix
- Result C is also upper triangular
- Uses merge-based approach for deterministic output

The symbolic phase computes:
1. How many non-zeros each row of C will have
2. Which columns will be non-zero (the pattern)

This allows us to allocate C's CSR arrays before the numeric phase.
"""

import numpy as np
from typing import Tuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# Maximum unique output columns per row
# This limits memory per thread, not the column index range
# Genealogy graphs typically have O(1) edges per node, so this is generous
MAX_UNIQUE_PER_ROW = 512


# CUDA kernel for symbolic phase with dynamic per-row accumulator
SYMBOLIC_MERGE_KERNEL = r'''
extern "C" {

// Maximum unique output columns per row (must match Python constant)
#define MAX_UNIQUE_PER_ROW 512

// Symbolic phase kernel - one row per thread
// Computes row_nnz[i] = number of non-zeros in row i of C = A @ A
// Uses a dynamic per-row accumulator keyed by actual column indices
__global__ void symbolic_phase_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    int* __restrict__ row_nnz,
    int* __restrict__ overflow_flag,
    int n_rows
) {
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_i >= n_rows) return;
    
    // For row i of C, we need to compute C[i,:] = A[i,:] @ A
    // This means for each non-zero A[i,k], we add row k of A to the result
    
    int row_start = indptrA[row_i];
    int row_end = indptrA[row_i + 1];
    int row_len = row_end - row_start;
    
    if (row_len == 0) {
        row_nnz[row_i] = 0;
        return;
    }
    
    // Use a dynamic array to track unique column indices
    // This is keyed by actual column indices, not limited by column index range
    int unique_cols[MAX_UNIQUE_PER_ROW];
    int num_unique = 0;
    
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
                // Check if column j is already in our unique set (linear search)
                bool found = false;
                for (int u = 0; u < num_unique; u++) {
                    if (unique_cols[u] == j) {
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // Add new unique column
                    if (num_unique < MAX_UNIQUE_PER_ROW) {
                        unique_cols[num_unique] = j;
                        num_unique++;
                    } else {
                        // Overflow - set flag and continue
                        atomicExch(overflow_flag, row_i + 1);  // Store row+1 (0 means no overflow)
                    }
                }
            }
        }
    }
    
    row_nnz[row_i] = num_unique;
}

} // extern "C"
'''


class SymbolicPhase:
    """
    Symbolic phase implementation for SpGEMM.
    
    Computes the sparsity pattern for C = A @ A (upper triangular).
    """
    
    def __init__(self):
        """Initialize and compile symbolic phase kernel."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")
        
        self._module = cp.RawModule(code=SYMBOLIC_MERGE_KERNEL)
        self._kernel = self._module.get_function('symbolic_phase_kernel')
    
    def compute_pattern(
        self,
        indptrA: 'cp.ndarray',
        indicesA: 'cp.ndarray',
        n_rows: int
    ) -> 'cp.ndarray':
        """
        Compute row non-zero counts for C = A @ A.
        
        Args:
            indptrA: Row pointers of A (int32 or int64)
            indicesA: Column indices of A (int32)
            n_rows: Number of rows in A
        
        Returns:
            row_nnz: Array of length n_rows with count of non-zeros per row
        
        Raises:
            RuntimeError: If any row exceeds MAX_UNIQUE_PER_ROW unique output columns
        """
        # Allocate output
        row_nnz = cp.zeros(n_rows, dtype=cp.int32)
        
        # Allocate overflow flag (0 = no overflow, row+1 = overflow at row)
        overflow_flag = cp.zeros(1, dtype=cp.int32)
        
        # Launch kernel - one thread per row
        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size
        
        # Convert indptr to int32 if needed (kernel expects int32)
        indptrA_i32 = cp.asarray(indptrA, dtype=cp.int32)
        indicesA_i32 = cp.asarray(indicesA, dtype=cp.int32)
        
        self._kernel(
            (grid_size,), (block_size,),
            (indptrA_i32, indicesA_i32, row_nnz, overflow_flag, n_rows)
        )
        
        # Check for overflow
        overflow_row = int(overflow_flag[0].get())
        if overflow_row > 0:
            raise RuntimeError(
                f"SpGEMM symbolic phase overflow: row {overflow_row - 1} exceeded "
                f"maximum of {MAX_UNIQUE_PER_ROW} unique output columns. "
                f"This typically indicates a very dense graph that exceeds "
                f"the expected genealogy workload characteristics."
            )
        
        return row_nnz


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
    # Compute per-row non-zero counts
    symbolic = SymbolicPhase()
    row_nnz = symbolic.compute_pattern(indptrA, indicesA, n_rows)
    
    # Build indptr via prefix sum
    indptrC = prefix_sum_scan(row_nnz)
    
    # Total nnz is the last element of indptr
    nnzC = int(indptrC[-1].get())
    
    return indptrC, nnzC


# Singleton instance
_symbolic = None


def get_symbolic_phase() -> SymbolicPhase:
    """Get or create singleton symbolic phase instance."""
    global _symbolic
    if _symbolic is None:
        _symbolic = SymbolicPhase()
    return _symbolic
