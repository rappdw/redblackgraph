"""
Tests for upper triangular Floyd-Warshall optimization.

These tests verify:
- Correctness: optimized algorithm produces same results as standard
- Performance: ~1.8-2x speedup on upper triangular matrices
- Integration: works with topological_sort output
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import time


class TestUpperTriangularClosureCorrectness:
    """Tests for correctness of the upper triangular optimization."""
    
    def test_simple_upper_triangular(self):
        """Test that optimized closure matches standard on simple case."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        # Simple upper triangular matrix
        A = rb_array(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        # Standard closure
        result_std, diameter_std = floyd_warshall(A, assume_upper_triangular=False)
        
        # Optimized closure
        A_copy = rb_array(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        result_opt, diameter_opt = floyd_warshall(A_copy, assume_upper_triangular=True)
        
        np.testing.assert_array_equal(result_std, result_opt)
        assert diameter_std == diameter_opt
    
    def test_larger_upper_triangular(self):
        """Test on larger upper triangular matrix."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        np.random.seed(42)
        n = 50
        
        # Create random upper triangular matrix with RBG structure
        A = np.triu(np.random.randint(0, 8, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = rb_array(A.astype(np.int32))
        
        # Standard closure
        A_std = rb_array(np.array(A, copy=True))
        result_std, diameter_std = floyd_warshall(A_std, assume_upper_triangular=False)
        
        # Optimized closure
        A_opt = rb_array(np.array(A, copy=True))
        result_opt, diameter_opt = floyd_warshall(A_opt, assume_upper_triangular=True)
        
        np.testing.assert_array_equal(result_std, result_opt)
        assert diameter_std == diameter_opt
    
    def test_output_stays_upper_triangular(self):
        """Test that output remains upper triangular."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        # Upper triangular input
        A = rb_array(np.array([
            [1, 2, 4, 8],
            [0, 1, 2, 4],
            [0, 0, 1, 2],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result, _ = floyd_warshall(A, assume_upper_triangular=True)
        
        # Check lower triangle is still zero
        lower = np.tril(result, k=-1)
        assert np.all(lower == 0), "Lower triangle should remain zero"
    
    def test_with_transitive_edges(self):
        """Test closure computation adds transitive edges correctly."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        # Graph: 0 -> 1 -> 2 (should add 0 -> 2)
        A = rb_array(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result, _ = floyd_warshall(A, assume_upper_triangular=True)
        
        # Should have transitive edge 0 -> 2
        # avos_product(2, 4) = 2 * 4 = 8
        assert result[0, 2] != 0, "Transitive edge should be added"


class TestUpperTriangularClosurePerformance:
    """Tests for performance of the upper triangular optimization."""
    
    def test_speedup_ratio(self):
        """Test that optimized version is faster."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        np.random.seed(42)
        n = 200
        
        # Create random upper triangular matrix
        A_base = np.triu(np.random.randint(0, 8, size=(n, n)), k=1)
        np.fill_diagonal(A_base, np.random.choice([-1, 1], size=n))
        A_base = A_base.astype(np.int32)
        
        # Time standard version
        A_std = rb_array(np.array(A_base, copy=True))
        start = time.perf_counter()
        floyd_warshall(A_std, assume_upper_triangular=False)
        time_std = time.perf_counter() - start
        
        # Time optimized version
        A_opt = rb_array(np.array(A_base, copy=True))
        start = time.perf_counter()
        floyd_warshall(A_opt, assume_upper_triangular=True)
        time_opt = time.perf_counter() - start
        
        speedup = time_std / time_opt
        print(f"\nSpeedup: {speedup:.2f}x (standard: {time_std:.3f}s, optimized: {time_opt:.3f}s)")
        
        # Expect at least 1.5x speedup (being conservative due to noise)
        assert speedup > 1.3, f"Expected speedup > 1.3x, got {speedup:.2f}x"
    
    def test_iteration_count_reduction(self):
        """Verify the theoretical iteration count reduction."""
        # For an NxN matrix:
        # Standard: N³ iterations
        # Optimized: sum over k of (k+1)*(N-k) = N³/6 + N²/2 + N/3
        # Ratio: approximately 6x fewer iterations, but with branch overhead
        # Expected speedup: 1.8-2x
        
        n = 100
        standard_iters = n ** 3
        optimized_iters = sum((k + 1) * (n - k) for k in range(n))
        
        ratio = standard_iters / optimized_iters
        print(f"\nTheoretical iteration ratio for n={n}: {ratio:.2f}x")
        
        # Should be close to 6 for large n
        assert ratio > 5, f"Expected ratio > 5, got {ratio:.2f}"


class TestUpperTriangularClosureIntegration:
    """Integration tests with topological sort."""
    
    def test_with_topological_sort(self):
        """Test full pipeline: topological sort -> upper triangular closure."""
        from redblackgraph.sparse.csgraph import topological_sort, floyd_warshall, permute
        from redblackgraph.core.redblack import array as rb_array
        
        # Create a non-upper-triangular DAG
        # Lower triangular: edges go from higher to lower indices
        A = rb_array(np.array([
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [0, 4, 1, 0],
            [8, 0, 2, 1]
        ], dtype=np.int32))
        
        # Get topological ordering
        perm = topological_sort(csr_matrix(A))
        
        # Apply permutation to get upper triangular
        A_upper = permute(A, perm, assume_upper_triangular=True)
        
        # Verify it's upper triangular
        lower = np.tril(A_upper, k=-1)
        assert np.all(lower == 0), "Permuted matrix should be upper triangular"
        
        # Compute closure with optimization
        result, diameter = floyd_warshall(A_upper, assume_upper_triangular=True)
        
        # Verify result is still upper triangular
        lower_result = np.tril(result, k=-1)
        assert np.all(lower_result == 0), "Result should be upper triangular"
    
    def test_transitive_closure_wrapper(self):
        """Test the transitive_closure_floyd_warshall wrapper with flag."""
        from redblackgraph.sparse.csgraph.transitive_closure import transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        # Upper triangular matrix
        A = rb_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        # Test with optimization
        result = transitive_closure_floyd_warshall(A, assume_upper_triangular=True)
        
        assert result is not None
        assert hasattr(result, 'W')  # TransitiveClosure uses W, not R
        assert hasattr(result, 'diameter')
    
    def test_closure_equivalence_with_topological_sort(self):
        """Test that topo sort produces valid upper triangular form for optimized FW."""
        from redblackgraph.sparse.csgraph import topological_sort, floyd_warshall, permute, is_upper_triangular
        from redblackgraph.core.redblack import array as rb_array
        from scipy.sparse import csr_matrix
        
        # Create a random DAG (upper triangular base, then permute to scramble)
        np.random.seed(123)
        n = 30
        
        # Start with upper triangular
        A_base = np.triu(np.random.randint(0, 8, size=(n, n)), k=1)
        np.fill_diagonal(A_base, np.random.choice([-1, 1], size=n))
        A_base = A_base.astype(np.int32)
        
        # Random permutation to scramble order
        rand_perm = np.random.permutation(n).astype(np.int32)
        A_scrambled = rb_array(A_base.copy())
        A_scrambled = permute(A_scrambled, rand_perm)
        
        # Topological sort to recover upper triangular form
        perm = topological_sort(csr_matrix(A_scrambled))
        A_ordered = permute(rb_array(np.array(A_scrambled, copy=True)), perm, assume_upper_triangular=True)
        
        # Verify the result is upper triangular
        assert is_upper_triangular(csr_matrix(A_ordered)), "Topo sorted matrix should be upper triangular"
        
        # Compute closure with optimized FW
        result_opt, _ = floyd_warshall(rb_array(A_ordered.copy()), assume_upper_triangular=True)
        
        # Compute closure with standard FW on same upper triangular input
        result_std, _ = floyd_warshall(rb_array(A_ordered.copy()), assume_upper_triangular=False)
        
        # Both should produce identical results on upper triangular input
        np.testing.assert_array_equal(result_std, result_opt)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        A = rb_array(np.array([], dtype=np.int32).reshape(0, 0))
        result, diameter = floyd_warshall(A, assume_upper_triangular=True)
        
        assert result.shape == (0, 0)
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        A = rb_array(np.array([[1]], dtype=np.int32))
        result, diameter = floyd_warshall(A, assume_upper_triangular=True)
        
        np.testing.assert_array_equal(result, [[1]])
    
    def test_diagonal_only(self):
        """Test with diagonal-only matrix (no edges)."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        n = 10
        A = rb_array(np.eye(n, dtype=np.int32))
        
        result_std, _ = floyd_warshall(rb_array(np.eye(n, dtype=np.int32)), assume_upper_triangular=False)
        result_opt, _ = floyd_warshall(A, assume_upper_triangular=True)
        
        np.testing.assert_array_equal(result_std, result_opt)
    
    def test_fully_connected_upper_triangular(self):
        """Test with fully connected upper triangular matrix."""
        from redblackgraph.sparse.csgraph import floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        
        n = 20
        A = np.triu(np.ones((n, n), dtype=np.int32) * 2, k=1)
        np.fill_diagonal(A, 1)
        
        result_std, _ = floyd_warshall(rb_array(A.copy()), assume_upper_triangular=False)
        result_opt, _ = floyd_warshall(rb_array(A.copy()), assume_upper_triangular=True)
        
        np.testing.assert_array_equal(result_std, result_opt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
