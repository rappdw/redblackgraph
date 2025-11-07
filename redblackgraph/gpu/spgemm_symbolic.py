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


# CUDA kernel for symbolic phase with merge-based approach
SYMBOLIC_MERGE_KERNEL = r'''
extern "C" {

// Merge two sorted arrays and count unique elements with upper triangular mask
// Returns count of unique elements where j >= row_idx
__device__ int merge_count_unique_masked(
    const int* __restrict__ list1,
    int len1,
    const int* __restrict__ list2,
    int len2,
    int row_idx
) {
    int i = 0, j = 0;
    int count = 0;
    int last = -1;  // Track last seen value to detect duplicates
    
    while (i < len1 || j < len2) {
        int val;
        
        // Get next value from whichever list has the smaller element
        if (i >= len1) {
            val = list2[j++];
        } else if (j >= len2) {
            val = list1[i++];
        } else if (list1[i] < list2[j]) {
            val = list1[i++];
        } else if (list1[i] > list2[j]) {
            val = list2[j++];
        } else {
            // Equal - take from either and advance both
            val = list1[i++];
            j++;
        }
        
        // Apply upper triangular mask: only count if val >= row_idx
        // Also skip if duplicate
        if (val >= row_idx && val != last) {
            count++;
            last = val;
        }
    }
    
    return count;
}

// Symbolic phase kernel - one row per thread
// Computes row_nnz[i] = number of non-zeros in row i of C = A @ A
__global__ void symbolic_phase_kernel(
    const int* __restrict__ indptrA,
    const int* __restrict__ indicesA,
    int* __restrict__ row_nnz,
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
    
    // For simplicity in this first version, we'll do a simple approach:
    // - Collect all candidate columns from all contributing rows
    // - Sort and unique them
    // - Apply triangular mask
    
    // This is a simplified version - production would use hash sets or
    // more sophisticated merging. For now, we'll handle small rows.
    
    // Count unique columns across all contributing rows
    int total_count = 0;
    
    // Use a simple bitmap for columns (limited to 1024 for this prototype)
    // In production, use hash tables or dynamic allocation
    const int MAX_COLS = 1024;
    bool seen[MAX_COLS];
    for (int i = 0; i < MAX_COLS; i++) {
        seen[i] = false;
    }
    
    // For each non-zero in row i of A
    for (int k_idx = row_start; k_idx < row_end; k_idx++) {
        int k = indicesA[k_idx];
        
        // Add columns from row k of A
        int k_row_start = indptrA[k];
        int k_row_end = indptrA[k + 1];
        
        for (int j_idx = k_row_start; j_idx < k_row_end; j_idx++) {
            int j = indicesA[j_idx];
            
            // Apply triangular mask: only j >= row_i
            if (j >= row_i && j < MAX_COLS) {
                if (!seen[j]) {
                    seen[j] = true;
                    total_count++;
                }
            }
        }
    }
    
    row_nnz[row_i] = total_count;
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
        """
        # Allocate output
        row_nnz = cp.zeros(n_rows, dtype=cp.int32)
        
        # Launch kernel - one thread per row
        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size
        
        # Convert indptr to int32 if needed (kernel expects int32)
        indptrA_i32 = cp.asarray(indptrA, dtype=cp.int32)
        indicesA_i32 = cp.asarray(indicesA, dtype=cp.int32)
        
        self._kernel(
            (grid_size,), (block_size,),
            (indptrA_i32, indicesA_i32, row_nnz, n_rows)
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
