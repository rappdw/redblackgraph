"""
Tests for adaptive transitive closure and repeated squaring.

These tests verify:
- transitive_closure_squaring produces correct results
- transitive_closure_adaptive selects appropriate algorithms
- All methods produce equivalent results
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix
import time


class TestTransitiveClosureSquaring:
    """Tests for repeated squaring closure."""
    
    def test_simple_chain(self):
        """Test on simple chain: 0 -> 1 -> 2."""
        from redblackgraph.sparse.csgraph import transitive_closure_squaring
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_squaring(A)
        W = result.W.toarray()
        
        # Should have transitive edge 0 -> 2
        assert W[0, 2] != 0
    
    def test_matches_floyd_warshall(self):
        """Test that squaring matches Floyd-Warshall."""
        from redblackgraph.sparse.csgraph import transitive_closure_squaring, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(42)
        n = 20
        
        # Create random upper triangular matrix
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Squaring
        result_sq = transitive_closure_squaring(csr_matrix(A))
        
        # Floyd-Warshall
        result_fw = transitive_closure_floyd_warshall(rb_matrix(A))
        
        # Should be equal
        np.testing.assert_array_equal(result_sq.W.toarray(), result_fw.W)
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import transitive_closure_squaring
        
        A = csr_matrix([[1]], dtype=np.int32)
        result = transitive_closure_squaring(A)
        
        assert result.W.toarray()[0, 0] == 1
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import transitive_closure_squaring
        
        A = csr_matrix((0, 0), dtype=np.int32)
        result = transitive_closure_squaring(A)
        
        assert result.W.shape == (0, 0)
    
    def test_stays_sparse(self):
        """Test that result is sparse."""
        from redblackgraph.sparse.csgraph import transitive_closure_squaring
        
        n = 50
        # Build chain matrix properly using COO constructor
        # (avoids scipy's in-place modification issues with CSR)
        row = list(range(n)) + list(range(n-1))  # diagonal + upper diagonal
        col = list(range(n)) + list(range(1, n))
        data = [1] * n + [2] * (n-1)
        A = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
        
        result = transitive_closure_squaring(A)
        
        assert isspmatrix(result.W)


class TestTransitiveClosureAdaptive:
    """Tests for adaptive closure strategy."""
    
    def test_selects_component_wise_for_disconnected(self):
        """Test that component-wise is used for disconnected graphs."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        # Two disconnected components
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A)
        
        # Result should be sparse (component-wise produces sparse)
        assert isspmatrix(result.W)
        
        # No edges between components
        W = result.W.toarray()
        assert W[0, 2] == 0
        assert W[0, 3] == 0
    
    def test_method_override_fw(self):
        """Test method override for Floyd-Warshall."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A, method="FW")
        expected = transitive_closure_floyd_warshall(rb_matrix(A.toarray()))
        
        np.testing.assert_array_equal(
            result.W.toarray() if isspmatrix(result.W) else result.W,
            expected.W
        )
    
    def test_method_override_dijkstra(self):
        """Test method override for Dijkstra."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive, transitive_closure_dijkstra
        from redblackgraph.sparse import rb_matrix
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A, method="D")
        expected = transitive_closure_dijkstra(rb_matrix(A.toarray()))
        
        np.testing.assert_array_equal(
            result.W.toarray() if isspmatrix(result.W) else result.W,
            expected.W
        )
    
    def test_method_override_squaring(self):
        """Test method override for squaring."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive, transitive_closure_squaring
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A, method="squaring")
        expected = transitive_closure_squaring(A)
        
        np.testing.assert_array_equal(
            result.W.toarray(),
            expected.W.toarray()
        )
    
    def test_upper_triangular_optimization(self):
        """Test that upper triangular matrices use optimized FW."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        # Upper triangular matrix
        A = csr_matrix(np.array([
            [1, 2, 4, 8],
            [0, 1, 2, 4],
            [0, 0, 1, 2],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A)
        
        # Should produce correct result
        W = result.W.toarray() if isspmatrix(result.W) else result.W
        assert W[0, 3] != 0  # Transitive edge should exist
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        A = csr_matrix((0, 0), dtype=np.int32)
        result = transitive_closure_adaptive(A)
        
        assert result.W.shape == (0, 0)
    
    def test_dense_input(self):
        """Test with dense input."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        A = np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32)
        
        result = transitive_closure_adaptive(A)
        
        # Should work with dense input
        assert result.W is not None


class TestAllMethodsEquivalent:
    """Tests that all methods produce equivalent results."""
    
    def test_all_methods_match(self):
        """Test that all closure methods produce same result."""
        from redblackgraph.sparse.csgraph import (
            transitive_closure_floyd_warshall,
            transitive_closure_dijkstra,
            transitive_closure_squaring,
            transitive_closure_adaptive
        )
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(123)
        n = 15
        
        # Create random upper triangular matrix
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Compute closure with each method
        result_fw = transitive_closure_floyd_warshall(rb_matrix(A))
        result_d = transitive_closure_dijkstra(rb_matrix(A))
        result_sq = transitive_closure_squaring(csr_matrix(A))
        result_adapt = transitive_closure_adaptive(csr_matrix(A))
        
        # All should match Floyd-Warshall reference
        W_fw = result_fw.W
        W_d = result_d.W
        W_sq = result_sq.W.toarray()
        W_adapt = result_adapt.W.toarray() if isspmatrix(result_adapt.W) else result_adapt.W
        
        np.testing.assert_array_equal(W_d, W_fw)
        np.testing.assert_array_equal(W_sq, W_fw)
        np.testing.assert_array_equal(W_adapt, W_fw)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_all_singletons(self):
        """Test with all isolated vertices."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        n = 20
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        result = transitive_closure_adaptive(A)
        
        # Should just be identity
        np.testing.assert_array_equal(
            result.W.toarray() if isspmatrix(result.W) else result.W,
            np.eye(n, dtype=np.int32)
        )
    
    def test_fully_connected(self):
        """Test with fully connected graph."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        n = 10
        A = csr_matrix(np.triu(np.ones((n, n), dtype=np.int32)))
        
        result = transitive_closure_adaptive(A)
        
        assert result.W is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
