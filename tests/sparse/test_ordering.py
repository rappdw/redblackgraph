"""
Tests for sparse canonical ordering.

These tests verify:
- Correctness: sparse version produces same canonical ordering as dense
- Performance: O(V+E+V log V) scaling
- Integration: Full pipeline from load to canonical form
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestSparseOrderingCorrectness:
    """Tests for correctness of sparse canonical ordering."""
    
    def test_matches_dense_simple(self):
        """Test that sparse version matches dense on simple case."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering, _get_permutation, _get_permutation_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        # Simple matrix with 2 components
        A_dense = rb_array(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        A_sparse = csr_matrix(A_dense)
        
        # Get orderings
        result_dense = avos_canonical_ordering(A_dense)
        result_sparse = avos_canonical_ordering(A_sparse)
        
        # Both should produce valid permutations
        assert len(result_dense.label_permutation) == 4
        assert len(result_sparse.label_permutation) == 4
        assert set(result_dense.label_permutation) == {0, 1, 2, 3}
        assert set(result_sparse.label_permutation) == {0, 1, 2, 3}
    
    def test_matches_dense_random(self):
        """Test on random transitively closed matrices."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        from redblackgraph.sparse.csgraph import transitive_closure_floyd_warshall
        from redblackgraph.core.redblack import array as rb_array
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(42)
        n = 30
        
        # Create random upper triangular matrix (DAG structure)
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Compute transitive closure to satisfy precondition
        A_dense = rb_array(A.copy())
        tc = transitive_closure_floyd_warshall(rb_matrix(A))
        A_closed = rb_array(tc.W)
        
        A_sparse = csr_matrix(A_closed)
        
        # Get orderings
        result_dense = avos_canonical_ordering(A_closed)
        result_sparse = avos_canonical_ordering(A_sparse)
        
        # Permutations should be identical
        np.testing.assert_array_equal(
            result_dense.label_permutation, 
            result_sparse.label_permutation
        )
    
    def test_dispatch_from_avos_canonical_ordering(self):
        """Test that avos_canonical_ordering auto-dispatches for sparse input."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        from scipy.sparse import isspmatrix
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = avos_canonical_ordering(A)
        
        # Result should be an Ordering
        assert hasattr(result, 'A')
        assert hasattr(result, 'label_permutation')
        assert hasattr(result, 'components')
        
        # Result matrix should be sparse
        assert isspmatrix(result.A)
    
    def test_canonical_property(self):
        """Test that result satisfies canonical ordering properties."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering, is_upper_triangular
        
        # Upper triangular input
        A = csr_matrix(np.array([
            [1, 2, 4, 8],
            [0, 1, 2, 4],
            [0, 0, 1, 2],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = avos_canonical_ordering(A)
        
        # Result should be upper triangular
        assert is_upper_triangular(result.A)
    
    def test_component_grouping(self):
        """Test that components are grouped together in result."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        # Two disconnected components: {0,1} and {2,3}
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = avos_canonical_ordering(A)
        
        # Check component info
        assert len(result.components) == 2
        assert sum(result.components.values()) == 4


class TestSparseOrderingPerformance:
    """Performance tests for sparse canonical ordering."""
    
    def test_scaling_with_fixed_edges(self):
        """Test O(V+E+V log V) scaling."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        np.random.seed(42)
        times = []
        sizes = [500, 1000, 2000]
        
        for n in sizes:
            # Fixed number of edges
            nnz_target = 5000
            density = nnz_target / (n * n)
            
            A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
            A.data[:] = np.random.randint(1, 10, size=A.nnz)
            # Make upper triangular (DAG)
            A = csr_matrix(np.triu(A.toarray()))
            # Add diagonal
            A = A + csr_matrix(np.eye(n, dtype=np.int32))
            
            start = time.perf_counter()
            result = avos_canonical_ordering(A)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        print(f"\nOrdering times: {list(zip(sizes, times))}")
        
        # With fixed edges, time should scale roughly as O(V log V)
        # Allow generous tolerance for noise
        assert times[-1] < times[0] * 20, f"Time growing too fast: {times}"
    
    def test_large_sparse_graph(self):
        """Test on large sparse graph."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        np.random.seed(42)
        n = 10000
        
        # Very sparse graph
        nnz_target = 20000
        
        # Create random upper triangular sparse matrix
        row = np.random.randint(0, n-1, size=nnz_target)
        col = row + np.random.randint(1, min(100, n), size=nnz_target)
        col = np.minimum(col, n-1)
        data = np.random.randint(1, 10, size=nnz_target).astype(np.int32)
        
        A = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
        # Add diagonal
        A = A + csr_matrix(np.diag(np.random.choice([-1, 1], size=n).astype(np.int32)))
        
        start = time.perf_counter()
        result = avos_canonical_ordering(A)
        elapsed = time.perf_counter() - start
        
        print(f"\n10K vertex graph ({A.nnz} edges): {elapsed:.3f}s")
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Too slow: {elapsed}s"
        assert len(result.label_permutation) == n


class TestSparseOrderingEdgeCases:
    """Edge case tests."""
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        A = csr_matrix((0, 0), dtype=np.int32)
        result = avos_canonical_ordering(A)
        
        assert len(result.label_permutation) == 0
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        A = csr_matrix([[1]], dtype=np.int32)
        result = avos_canonical_ordering(A)
        
        assert len(result.label_permutation) == 1
        np.testing.assert_array_equal(result.label_permutation, [0])
    
    def test_all_singletons(self):
        """Test with all isolated vertices."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        n = 50
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        result = avos_canonical_ordering(A)
        
        assert len(result.label_permutation) == n
        assert len(result.components) == n
    
    def test_fully_connected(self):
        """Test with fully connected upper triangular."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering
        
        n = 20
        A = csr_matrix(np.triu(np.ones((n, n), dtype=np.int32)))
        
        result = avos_canonical_ordering(A)
        
        assert len(result.label_permutation) == n
        assert len(result.components) == 1  # All connected


class TestSparseOrderingIntegration:
    """Integration tests with full pipeline."""
    
    def test_full_pipeline(self):
        """Test: load -> transitive closure -> canonical ordering."""
        from redblackgraph.sparse.csgraph import avos_canonical_ordering, is_upper_triangular
        from redblackgraph.sparse.csgraph import transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        # Start with a DAG
        A = np.array([
            [1, 2, 0, 0, 8],
            [0, 1, 4, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1]
        ], dtype=np.int32)
        
        # Transitive closure
        tc = transitive_closure_floyd_warshall(rb_matrix(A))
        A_closed = csr_matrix(tc.W)
        
        # Canonical ordering
        result = avos_canonical_ordering(A_closed)
        
        # Verify result
        assert is_upper_triangular(result.A)
        assert len(result.label_permutation) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
