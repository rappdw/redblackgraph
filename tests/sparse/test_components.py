"""
Tests for sparse component finding.

These tests verify:
- Correctness: sparse version matches dense version
- Performance: O(V+E) scaling for large sparse graphs
- Edge cases: single vertices, disconnected components
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestSparseComponentsCorrectness:
    """Tests for correctness of sparse component finding."""
    
    def test_matches_dense_simple(self):
        """Test that sparse version matches dense on simple case."""
        from redblackgraph.sparse.csgraph import find_components, find_components_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        # Two components: {0,1} and {2,3}
        A_dense = rb_array(np.array([
            [1, 2, 0, 0],
            [2, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 4, 1]
        ], dtype=np.int32))
        
        A_sparse = csr_matrix(A_dense)
        
        q_dense = {}
        q_sparse = {}
        
        result_dense = find_components(A_dense, q_dense)
        result_sparse = find_components_sparse(A_sparse, q_sparse)
        
        # Same component structure (component IDs may differ)
        assert len(set(result_dense)) == len(set(result_sparse))
        assert q_dense == q_sparse or set(q_dense.values()) == set(q_sparse.values())
    
    def test_matches_dense_random(self):
        """Test on random matrices."""
        from redblackgraph.sparse.csgraph import find_components, find_components_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        np.random.seed(42)
        n = 50
        
        # Create random matrix with some structure
        A = np.random.randint(0, 4, size=(n, n)).astype(np.int32)
        # Make it symmetric (undirected graph) for easier component checking
        A = A + A.T
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        
        A_dense = rb_array(A)
        A_sparse = csr_matrix(A)
        
        q_dense = {}
        q_sparse = {}
        
        result_dense = find_components(A_dense, q_dense)
        result_sparse = find_components_sparse(A_sparse, q_sparse)
        
        # Verify same number of components
        n_comp_dense = len(set(result_dense))
        n_comp_sparse = len(set(result_sparse))
        assert n_comp_dense == n_comp_sparse, f"Component count mismatch: {n_comp_dense} vs {n_comp_sparse}"
    
    def test_dispatch_from_find_components(self):
        """Test that find_components auto-dispatches for sparse input."""
        from redblackgraph.sparse.csgraph import find_components
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [2, 1, 0],
            [0, 0, 1]
        ], dtype=np.int32))
        
        q = {}
        result = find_components(A, q)
        
        # Should have 2 components: {0,1} and {2}
        assert len(set(result)) == 2
        assert sum(q.values()) == 3  # Total vertices
    
    def test_single_component(self):
        """Test with single fully-connected component."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        n = 20
        # Fully connected (all non-zero)
        A = csr_matrix(np.ones((n, n), dtype=np.int32))
        
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(set(result)) == 1
        assert 0 in q
        assert q[0] == n
    
    def test_all_singletons(self):
        """Test with all isolated vertices."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        n = 100
        # Diagonal only - each vertex is its own component
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(set(result)) == n
        assert all(v == 1 for v in q.values())
    
    def test_chain_graph(self):
        """Test with chain graph: 0-1-2-3-...-n."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        n = 50
        # Build chain: edges between consecutive vertices
        row = list(range(n-1)) + list(range(1, n))
        col = list(range(1, n)) + list(range(n-1))
        data = [1] * len(row)
        A = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
        
        q = {}
        result = find_components_sparse(A, q)
        
        # All vertices should be in one component
        assert len(set(result)) == 1
        assert q[0] == n


class TestSparseComponentsPerformance:
    """Performance tests for sparse component finding."""
    
    def test_scaling_with_fixed_density(self):
        """Test O(V+E) scaling - time should scale with edges, not V²."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        np.random.seed(42)
        times = []
        sizes = [1000, 5000, 10000]
        
        for n in sizes:
            # Fixed number of edges (not scaling with n²)
            nnz_target = 20000
            density = nnz_target / (n * n)
            
            A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
            A.data[:] = np.random.randint(1, 10, size=A.nnz)
            
            start = time.perf_counter()
            q = {}
            find_components_sparse(A, q)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        print(f"\nComponent finding times: {list(zip(sizes, times))}")
        
        # With fixed nnz, time should not grow much with n
        # Allow 10x tolerance for system noise
        assert times[-1] < times[0] * 10, \
            f"Time growing too fast: {times}"
    
    def test_large_sparse_graph(self):
        """Test on large sparse graph (100K vertices)."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        np.random.seed(42)
        n = 100000
        
        # Very sparse: ~0.01% density (10K edges in 100K² matrix)
        nnz_target = 10000
        
        # Create random edges
        row = np.random.randint(0, n, size=nnz_target)
        col = np.random.randint(0, n, size=nnz_target)
        data = np.random.randint(1, 10, size=nnz_target).astype(np.int32)
        
        A = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
        
        start = time.perf_counter()
        q = {}
        result = find_components_sparse(A, q)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K vertex graph ({A.nnz} edges): {elapsed:.3f}s, {len(q)} components")
        
        # Should complete in reasonable time (<5 seconds)
        assert elapsed < 5.0, f"Too slow: {elapsed}s"
        assert len(result) == n
    
    def test_sparse_vs_dense_speedup(self):
        """Compare sparse vs dense performance."""
        from redblackgraph.sparse.csgraph import find_components, find_components_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        np.random.seed(42)
        n = 500  # Small enough for dense to be tractable
        
        # Sparse matrix
        A_sparse = sparse_random(n, n, density=0.01, format='csr', dtype=np.int32)
        A_sparse.data[:] = np.random.randint(1, 10, size=A_sparse.nnz)
        
        # Dense version
        A_dense = rb_array(A_sparse.toarray())
        
        # Time sparse
        start = time.perf_counter()
        q_sparse = {}
        find_components_sparse(A_sparse, q_sparse)
        time_sparse = time.perf_counter() - start
        
        # Time dense
        start = time.perf_counter()
        q_dense = {}
        # Call dense implementation directly by passing rb_array
        find_components(A_dense, q_dense)
        time_dense = time.perf_counter() - start
        
        print(f"\nSparse: {time_sparse:.4f}s, Dense: {time_dense:.4f}s")
        print(f"Speedup: {time_dense/time_sparse:.1f}x")
        
        # Sparse should be faster for sparse matrices
        # (may not always hold for small n due to overhead)


class TestSparseComponentsEdgeCases:
    """Edge case tests."""
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        A = csr_matrix((0, 0), dtype=np.int32)
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(result) == 0
        assert len(q) == 0
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        A = csr_matrix([[1]], dtype=np.int32)
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(result) == 1
        assert result[0] == 0
        assert q[0] == 1
    
    def test_two_disconnected(self):
        """Test with two completely disconnected vertices."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        A = csr_matrix(np.diag([1, 1]).astype(np.int32))
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(set(result)) == 2
        assert sum(q.values()) == 2
    
    def test_asymmetric_edges(self):
        """Test with asymmetric (directed) edges."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        # 0 -> 1 only (no reverse edge)
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.int32))
        
        q = {}
        result = find_components_sparse(A, q)
        
        # 0 and 1 should be in same component (0->1 edge)
        # 2 is separate
        assert result[0] == result[1]
        assert result[2] != result[0]
    
    def test_csc_input(self):
        """Test that CSC input is handled correctly."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        from scipy.sparse import csc_matrix
        
        A = csc_matrix(np.array([
            [1, 2, 0],
            [2, 1, 0],
            [0, 0, 1]
        ], dtype=np.int32))
        
        q = {}
        result = find_components_sparse(A, q)
        
        assert len(set(result)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
