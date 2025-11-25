"""
Tests for sparse matrix permutation.

These tests verify that permute_sparse maintains O(nnz) complexity
and does NOT densify the matrix.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestSparsePermutation:
    """Tests for sparse permutation functionality."""
    
    def test_permute_sparse_basic(self):
        """Test basic sparse permutation."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Create a simple sparse matrix
        # [1, 2, 0]
        # [0, 3, 0]
        # [4, 0, 5]
        data = [1, 2, 3, 4, 5]
        row = [0, 0, 1, 2, 2]
        col = [0, 1, 1, 0, 2]
        A = csr_matrix((data, (row, col)), shape=(3, 3), dtype=np.int32)
        
        # Permutation: swap rows/cols 0 and 2
        p = np.array([2, 1, 0], dtype=np.int32)
        
        B = permute_sparse(A, p)
        
        # Expected result:
        # Row 0 of B = Row 2 of A with columns remapped
        # B[0,0] = A[2,2] = 5
        # B[0,2] = A[2,0] = 4
        # etc.
        expected = np.array([
            [5, 0, 4],
            [0, 3, 0],
            [0, 2, 1]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(B.toarray(), expected)
    
    def test_permute_sparse_preserves_nnz(self):
        """Test that sparse permutation preserves nnz count."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        n = 100
        A = sparse_random(n, n, density=0.05, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)  # Replace with integers
        
        p = np.random.permutation(n).astype(np.int32)
        
        B = permute_sparse(A, p)
        
        assert B.nnz == A.nnz, "Permutation should preserve nnz"
    
    def test_permute_sparse_no_densification(self):
        """Test that sparse permutation does NOT densify."""
        from redblackgraph.sparse.csgraph import permute_sparse, get_density
        
        n = 1000
        A = sparse_random(n, n, density=0.01, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)
        
        original_density = get_density(A)
        
        p = np.random.permutation(n).astype(np.int32)
        B = permute_sparse(A, p)
        
        result_density = get_density(B)
        
        assert abs(result_density - original_density) < 1e-10, \
            f"Density changed: {original_density} -> {result_density}"
    
    def test_permute_sparse_upper_triangular(self):
        """Test sparse permutation with upper triangular filtering."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Create matrix
        A = csr_matrix(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32))
        
        # Identity permutation with upper triangular flag
        p = np.array([0, 1, 2], dtype=np.int32)
        
        B = permute_sparse(A, p, assume_upper_triangular=True)
        
        # Should only keep upper triangular entries
        expected = np.array([
            [1, 2, 3],
            [0, 5, 6],
            [0, 0, 9]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(B.toarray(), expected)
    
    def test_permute_sparse_matches_dense(self):
        """Test that sparse permutation matches dense permutation result."""
        from redblackgraph.sparse.csgraph import permute, permute_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        # Create a test matrix
        dense = np.array([
            [-1, 2, 0, 4],
            [0, 1, 0, 0],
            [0, 6, -1, 8],
            [0, 0, 0, 1]
        ], dtype=np.int32)
        
        sparse = csr_matrix(dense)
        p = np.array([3, 1, 0, 2], dtype=np.int32)
        
        # Dense permutation (using rb_array for compatibility)
        dense_result = permute(rb_array(dense), p)
        
        # Sparse permutation
        sparse_result = permute_sparse(sparse, p)
        
        np.testing.assert_array_equal(sparse_result.toarray(), dense_result)
    
    def test_permute_dispatch(self):
        """Test that permute() auto-dispatches based on input type."""
        from redblackgraph.sparse.csgraph import permute
        from redblackgraph.core.redblack import array as rb_array
        
        # Dense input
        dense = rb_array(np.eye(5, dtype=np.int32))
        p = np.array([4, 3, 2, 1, 0], dtype=np.int32)
        
        dense_result = permute(dense, p)
        assert isinstance(dense_result, np.ndarray)
        
        # Sparse input
        sparse = csr_matrix(np.eye(5, dtype=np.int32))
        sparse_result = permute(sparse, p)
        assert hasattr(sparse_result, 'toarray')  # Is sparse
    
    def test_permute_sparse_complexity(self):
        """Test O(nnz) complexity of sparse permutation."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Fixed nnz, varying n
        times = []
        for n in [1000, 5000, 10000]:
            nnz_target = 5000
            density = nnz_target / (n * n)
            A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
            A.data[:] = np.random.randint(1, 100, size=A.nnz)
            
            p = np.random.permutation(n).astype(np.int32)
            
            start = time.perf_counter()
            B = permute_sparse(A, p)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Time should not grow significantly with n (O(nnz) not O(n²))
        # Allow 10x tolerance for noise
        assert times[-1] < times[0] * 10, \
            f"Permutation time growing too fast: {times}"


class TestPermutationCorrectness:
    """Additional correctness tests for permutation."""
    
    def test_identity_permutation(self):
        """Test that identity permutation is a no-op."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        A = sparse_random(50, 50, density=0.1, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)
        
        p = np.arange(50, dtype=np.int32)
        B = permute_sparse(A, p)
        
        np.testing.assert_array_equal(A.toarray(), B.toarray())
    
    def test_double_permutation_inverse(self):
        """Test that applying permutation twice with inverse gives original."""
        from redblackgraph.sparse.csgraph import permute_sparse
        
        n = 30
        A = sparse_random(n, n, density=0.1, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)
        
        p = np.random.permutation(n).astype(np.int32)
        p_inv = np.argsort(p).astype(np.int32)
        
        B = permute_sparse(A, p)
        C = permute_sparse(B, p_inv)
        
        np.testing.assert_array_equal(A.toarray(), C.toarray())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
