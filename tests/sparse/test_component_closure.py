"""
Tests for component-wise transitive closure.

These tests verify:
- Correctness: component-wise closure matches full closure
- Memory efficiency: sparse result for multi-component graphs
- Performance: faster than full graph closure for many components
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix
import time


class TestComponentWiseClosureCorrectness:
    """Tests for correctness of component-wise closure."""
    
    def test_two_components_simple(self):
        """Test on simple two-component graph."""
        from redblackgraph.sparse.csgraph import component_wise_closure, transitive_closure
        
        # Two disconnected DAGs
        # Component 1: 0 -> 1
        # Component 2: 2 -> 3
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = component_wise_closure(A)
        
        # Check result is sparse
        assert isspmatrix(result.W)
        
        # Check no edges between components
        W = result.W.toarray()
        assert W[0, 2] == 0
        assert W[0, 3] == 0
        assert W[1, 2] == 0
        assert W[1, 3] == 0
        assert W[2, 0] == 0
        assert W[2, 1] == 0
        assert W[3, 0] == 0
        assert W[3, 1] == 0
    
    def test_matches_full_closure(self):
        """Test that result matches full transitive closure."""
        from redblackgraph.sparse.csgraph import component_wise_closure, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        # Multi-component graph
        A = np.array([
            [1, 2, 0, 0, 0, 0],
            [0, 1, 4, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.int32)
        
        # Component-wise closure
        result_cw = component_wise_closure(csr_matrix(A))
        
        # Full closure
        result_full = transitive_closure_floyd_warshall(rb_matrix(A))
        
        # Should be equal
        np.testing.assert_array_equal(result_cw.W.toarray(), result_full.W)
    
    def test_single_component_fallback(self):
        """Test that single component uses standard closure."""
        from redblackgraph.sparse.csgraph import component_wise_closure, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        # Fully connected (single component)
        A = np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32)
        
        result_cw = component_wise_closure(csr_matrix(A))
        result_full = transitive_closure_floyd_warshall(rb_matrix(A))
        
        np.testing.assert_array_equal(result_cw.W.toarray(), result_full.W)
    
    def test_many_singletons(self):
        """Test with many isolated vertices."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        n = 100
        # All isolated vertices
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        result = component_wise_closure(A)
        
        # Should just be identity
        np.testing.assert_array_equal(result.W.toarray(), np.eye(n, dtype=np.int32))
    
    def test_transitive_edges_added(self):
        """Test that transitive edges are properly computed per component."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        # Chain: 0 -> 1 -> 2 (should add 0 -> 2)
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = component_wise_closure(A)
        W = result.W.toarray()
        
        # Should have transitive edge 0 -> 2
        assert W[0, 2] != 0


class TestComponentWiseClosureMemory:
    """Tests for memory efficiency."""
    
    def test_sparse_result(self):
        """Test that result is sparse."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        # 4 disconnected components
        n = 100
        A = csr_matrix(np.diag(np.ones(n, dtype=np.int32)))
        # Add some edges within first 25 vertices (component 0)
        for i in range(24):
            A[i, i+1] = 2
        
        result = component_wise_closure(A)
        
        assert isspmatrix(result.W)
        # Should be much sparser than n²
        assert result.W.nnz < n * n / 2
    
    def test_nnz_preserved_between_components(self):
        """Test that no edges are added between components."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        # Two components with internal edges
        A = csr_matrix(np.array([
            [1, 2, 4, 0, 0],  # Component 0
            [0, 1, 2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 8],  # Component 1
            [0, 0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = component_wise_closure(A)
        W = result.W.toarray()
        
        # No edges between components
        assert np.all(W[:3, 3:] == 0)
        assert np.all(W[3:, :3] == 0)


class TestComponentWiseClosurePerformance:
    """Performance tests."""
    
    def test_faster_than_full_for_many_components(self):
        """Test that component-wise is faster for many small components."""
        from redblackgraph.sparse.csgraph import component_wise_closure, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(42)
        n = 200  # 200 vertices
        n_components = 20  # 20 components of ~10 vertices each
        comp_size = n // n_components
        
        # Build block diagonal matrix
        blocks = []
        for _ in range(n_components):
            block = np.triu(np.random.randint(0, 4, size=(comp_size, comp_size)))
            np.fill_diagonal(block, 1)
            blocks.append(block)
        
        A = np.zeros((n, n), dtype=np.int32)
        for i, block in enumerate(blocks):
            start = i * comp_size
            end = start + comp_size
            A[start:end, start:end] = block
        
        # Time component-wise
        start = time.perf_counter()
        result_cw = component_wise_closure(csr_matrix(A))
        time_cw = time.perf_counter() - start
        
        # Time full closure
        start = time.perf_counter()
        result_full = transitive_closure_floyd_warshall(rb_matrix(A))
        time_full = time.perf_counter() - start
        
        print(f"\nComponent-wise: {time_cw:.4f}s, Full: {time_full:.4f}s")
        print(f"Speedup: {time_full/time_cw:.1f}x")
        
        # Verify correctness
        np.testing.assert_array_equal(result_cw.W.toarray(), result_full.W)


class TestComponentWiseClosureEdgeCases:
    """Edge case tests."""
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        A = csr_matrix((0, 0), dtype=np.int32)
        result = component_wise_closure(A)
        
        assert result.W.shape == (0, 0)
        assert result.diameter == 0
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        A = csr_matrix([[1]], dtype=np.int32)
        result = component_wise_closure(A)
        
        assert result.W.shape == (1, 1)
        assert result.W.toarray()[0, 0] == 1
    
    def test_method_fw(self):
        """Test with explicit FW method."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = component_wise_closure(A, method="FW")
        assert isspmatrix(result.W)
    
    def test_method_dijkstra(self):
        """Test with explicit Dijkstra method."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = component_wise_closure(A, method="D")
        assert isspmatrix(result.W)
    
    def test_dense_input(self):
        """Test with dense input (should be converted)."""
        from redblackgraph.sparse.csgraph import component_wise_closure
        
        A = np.array([
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.int32)
        
        result = component_wise_closure(A)
        assert isspmatrix(result.W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
