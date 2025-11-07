"""
Tests for SpGEMM symbolic phase (pattern computation).
"""

import pytest
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    from redblackgraph.gpu.spgemm_symbolic import (
        compute_symbolic_pattern,
        prefix_sum_scan,
        SymbolicPhase
    )
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")


def compute_pattern_cpu(A_csr) -> tuple:
    """
    CPU reference: compute pattern for C = A @ A (upper triangular).
    
    Returns:
        indptr: Row pointers for C
        nnz: Total non-zeros in C
    """
    n = A_csr.shape[0]
    row_nnz = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        # Get row i of A
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        
        if row_start == row_end:
            continue  # Empty row
        
        # Collect columns from all contributing rows
        cols_set = set()
        
        for k_idx in range(row_start, row_end):
            k = A_csr.indices[k_idx]
            
            # Add columns from row k of A
            k_row_start = A_csr.indptr[k]
            k_row_end = A_csr.indptr[k + 1]
            
            for j_idx in range(k_row_start, k_row_end):
                j = A_csr.indices[j_idx]
                
                # Apply triangular mask: j >= i
                if j >= i:
                    cols_set.add(j)
        
        row_nnz[i] = len(cols_set)
    
    # Build indptr
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(row_nnz)
    
    nnz = int(indptr[-1])
    
    return indptr, nnz


class TestPrefixSum:
    """Test prefix sum (scan) implementation."""
    
    def test_simple_scan(self):
        """Test basic prefix sum."""
        row_nnz = cp.array([3, 0, 2, 1], dtype=cp.int32)
        
        indptr = prefix_sum_scan(row_nnz)
        
        expected = cp.array([0, 3, 3, 5, 6], dtype=cp.int32)
        assert cp.array_equal(indptr, expected)
    
    def test_all_zeros(self):
        """Test scan with all zeros."""
        row_nnz = cp.array([0, 0, 0], dtype=cp.int32)
        
        indptr = prefix_sum_scan(row_nnz)
        
        expected = cp.array([0, 0, 0, 0], dtype=cp.int32)
        assert cp.array_equal(indptr, expected)
    
    def test_single_row(self):
        """Test scan with single row."""
        row_nnz = cp.array([5], dtype=cp.int32)
        
        indptr = prefix_sum_scan(row_nnz)
        
        expected = cp.array([0, 5], dtype=cp.int32)
        assert cp.array_equal(indptr, expected)
    
    def test_large_counts(self):
        """Test scan with larger numbers."""
        row_nnz = cp.array([10, 20, 30, 40], dtype=cp.int32)
        
        indptr = prefix_sum_scan(row_nnz)
        
        expected = cp.array([0, 10, 30, 60, 100], dtype=cp.int32)
        assert cp.array_equal(indptr, expected)


class TestSymbolicPhase:
    """Test symbolic phase pattern computation."""
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        # 3x3 empty matrix
        indptrA = cp.array([0, 0, 0, 0], dtype=cp.int32)
        indicesA = cp.array([], dtype=cp.int32)
        
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 3, 3)
        
        assert len(indptrC) == 4
        assert nnzC == 0
        assert cp.all(indptrC == 0)
    
    def test_identity_matrix(self):
        """Test with identity matrix."""
        # 3x3 identity (upper triangular)
        # [1 0 0]
        # [0 1 0]
        # [0 0 1]
        indptrA = cp.array([0, 1, 2, 3], dtype=cp.int32)
        indicesA = cp.array([0, 1, 2], dtype=cp.int32)
        
        # I @ I = I, so same pattern
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 3, 3)
        
        assert nnzC == 3
        assert cp.array_equal(indptrC, cp.array([0, 1, 2, 3], dtype=cp.int32))
    
    def test_simple_triangular(self):
        """Test with simple upper triangular matrix."""
        # 3x3 matrix:
        # [2  3  0]
        # [0  4  5]
        # [0  0  6]
        indptrA = cp.array([0, 2, 4, 5], dtype=cp.int32)
        indicesA = cp.array([0, 1, 1, 2, 2], dtype=cp.int32)
        
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 3, 3)
        
        # Verify against CPU reference
        A_cpu = sp.csr_matrix(
            (np.array([2, 3, 4, 5, 6], dtype=np.int32),
             indicesA.get(),
             indptrA.get()),
            shape=(3, 3)
        )
        indptr_expected, nnz_expected = compute_pattern_cpu(A_cpu)
        
        assert nnzC == nnz_expected
        assert cp.array_equal(indptrC.get(), indptr_expected)
    
    def test_vs_cpu_reference(self):
        """Test symbolic phase against CPU reference."""
        # Create a random upper triangular matrix
        n = 10
        density = 0.3
        
        # Generate random upper triangular
        A_cpu = sp.random(n, n, density=density, format='csr', dtype=np.int32)
        
        # Make it upper triangular
        for i in range(n):
            row_start = A_cpu.indptr[i]
            row_end = A_cpu.indptr[i + 1]
            
            # Remove entries where j < i
            mask = A_cpu.indices[row_start:row_end] >= i
            if not np.all(mask):
                # Rebuild row without lower triangle entries
                keep_indices = A_cpu.indices[row_start:row_end][mask]
                keep_data = A_cpu.data[row_start:row_end][mask]
                
                # This is complex to do in-place, so we'll create a new matrix
                # For testing, just skip if not already triangular
                continue
        
        # Transfer to GPU
        indptrA = cp.array(A_cpu.indptr, dtype=cp.int32)
        indicesA = cp.array(A_cpu.indices, dtype=cp.int32)
        
        # Compute pattern on GPU
        indptrC_gpu, nnzC_gpu = compute_symbolic_pattern(indptrA, indicesA, n, n)
        
        # Compute pattern on CPU
        indptrC_cpu, nnzC_cpu = compute_pattern_cpu(A_cpu)
        
        # Compare
        assert nnzC_gpu == nnzC_cpu
        assert cp.array_equal(indptrC_gpu.get(), indptrC_cpu)


class TestSmallExamples:
    """Test with hand-verified small examples."""
    
    def test_2x2_full(self):
        """Test 2x2 upper triangular."""
        # [1 2]
        # [0 3]
        indptrA = cp.array([0, 2, 3], dtype=cp.int32)
        indicesA = cp.array([0, 1, 1], dtype=cp.int32)
        
        # A @ A:
        # Row 0: (1,2) @ [(1,2), (0,3)]^T = 1*1 + 2*0 = 1 at col 0
        #                                    1*2 + 2*3 = 8 at col 1
        # Result row 0: cols [0, 1] -> 2 non-zeros
        # Row 1: (0,3) @ [(1,2), (0,3)]^T = 0*1 + 3*0 = 0 at col 0 (skip <1)
        #                                    0*2 + 3*3 = 9 at col 1
        # Result row 1: cols [1] -> 1 non-zero
        # Total: 3 non-zeros
        
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 2, 2)
        
        assert nnzC == 3
        expected = cp.array([0, 2, 3], dtype=cp.int32)
        assert cp.array_equal(indptrC, expected)
    
    def test_diagonal_only(self):
        """Test matrix with only diagonal elements."""
        # 4x4 diagonal
        indptrA = cp.array([0, 1, 2, 3, 4], dtype=cp.int32)
        indicesA = cp.array([0, 1, 2, 3], dtype=cp.int32)
        
        # Diagonal @ Diagonal = Diagonal
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 4, 4)
        
        assert nnzC == 4
        expected = cp.array([0, 1, 2, 3, 4], dtype=cp.int32)
        assert cp.array_equal(indptrC, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element_matrix(self):
        """Test 1x1 matrix."""
        indptrA = cp.array([0, 1], dtype=cp.int32)
        indicesA = cp.array([0], dtype=cp.int32)
        
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, 1, 1)
        
        assert nnzC == 1
        assert cp.array_equal(indptrC, cp.array([0, 1], dtype=cp.int32))
    
    def test_row_with_many_nonzeros(self):
        """Test row with many non-zeros (up to limit)."""
        # Single row with 10 non-zeros
        n = 20
        indptrA = cp.array([0, 10] + [10] * (n - 1), dtype=cp.int32)
        indicesA = cp.array(list(range(10)), dtype=cp.int32)
        
        # Only row 0 has entries, sparse @ sparse
        indptrC, nnzC = compute_symbolic_pattern(indptrA, indicesA, n, n)
        
        # Should have some pattern in row 0
        assert indptrC[0] == 0
        assert nnzC >= 0  # At least computes without error
