"""
Tests for sparse component finding and submatrix extraction.

These tests verify that:
1. find_components_sparse runs in O(V+E) time
2. extract_submatrix extracts correct subgraphs
3. merge_component_matrices correctly reconstructs full graphs
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestFindComponentsSparse:
    """Tests for sparse component finding."""
    
    def test_find_components_sparse_basic(self):
        """Test basic sparse component finding."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        # Create matrix with 2 disconnected components
        # Component 0: vertices 0, 1 (connected)
        # Component 1: vertices 2, 3 (connected)
        data = [1, 1, 1, 1]
        row = [0, 1, 2, 3]
        col = [1, 0, 3, 2]
        A = csr_matrix((data, (row, col)), shape=(4, 4), dtype=np.int32)
        
        q = {}
        components = find_components_sparse(A, q)
        
        # Should have 2 components
        assert len(set(components)) == 2
        
        # Vertices in same component should have same label
        assert components[0] == components[1]
        assert components[2] == components[3]
        
        # Verify component sizes
        assert q[components[0]] == 2
        assert q[components[2]] == 2
    
    def test_find_components_sparse_single_vertex(self):
        """Test component finding with isolated vertices."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        # 5 isolated vertices (no edges)
        A = csr_matrix((5, 5), dtype=np.int32)
        
        q = {}
        components = find_components_sparse(A, q)
        
        # Each vertex is its own component
        assert len(set(components)) == 5
        assert all(q[c] == 1 for c in q.values())
    
    def test_find_components_sparse_fully_connected(self):
        """Test component finding with fully connected graph."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        # Fully connected 5x5
        A = csr_matrix(np.ones((5, 5), dtype=np.int32))
        
        q = {}
        components = find_components_sparse(A, q)
        
        # All vertices in one component
        assert len(set(components)) == 1
        assert q[components[0]] == 5
    
    def test_find_components_sparse_matches_dense(self):
        """Test that sparse component finding matches dense version."""
        from redblackgraph.sparse.csgraph import find_components, find_components_sparse
        from redblackgraph.core.redblack import array as rb_array
        
        # Create a matrix with multiple components
        dense = np.array([
            [1, 2, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 4, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.int32)
        
        sparse = csr_matrix(dense)
        
        q_dense = {}
        components_dense = find_components(rb_array(dense), q_dense)
        
        q_sparse = {}
        components_sparse = find_components_sparse(sparse, q_sparse)
        
        # Should find same number of components
        assert len(set(components_dense)) == len(set(components_sparse))
        
        # Vertices should be grouped the same way
        # (component IDs may differ but grouping should match)
        for i in range(len(dense)):
            for j in range(len(dense)):
                dense_same = (components_dense[i] == components_dense[j])
                sparse_same = (components_sparse[i] == components_sparse[j])
                assert dense_same == sparse_same, \
                    f"Mismatch at ({i}, {j}): dense={dense_same}, sparse={sparse_same}"
    
    def test_find_components_sparse_complexity(self):
        """Test O(V+E) complexity of sparse component finding."""
        from redblackgraph.sparse.csgraph import find_components_sparse
        
        # Fixed edge count, varying vertex count
        times = []
        edge_count = 5000
        
        for n in [1000, 5000, 10000]:
            density = edge_count / (n * n)
            A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
            
            start = time.perf_counter()
            find_components_sparse(A)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Time should scale with V+E, not V²
        # Since E is fixed, time should scale roughly linearly with V
        # But building CSC adds some overhead, so allow generous tolerance
        assert times[-1] < times[0] * 20, \
            f"Component finding time growing too fast: {times}"


class TestExtractSubmatrix:
    """Tests for submatrix extraction."""
    
    def test_extract_submatrix_basic(self):
        """Test basic submatrix extraction."""
        from redblackgraph.sparse.csgraph import extract_submatrix
        
        # Full matrix
        data = [1, 2, 3, 4, 5, 6]
        row = [0, 0, 1, 1, 2, 2]
        col = [0, 2, 1, 2, 0, 2]
        A = csr_matrix((data, (row, col)), shape=(3, 3), dtype=np.int32)
        
        # Extract vertices 0 and 2
        vertices = np.array([0, 2], dtype=np.int32)
        submat, mapping = extract_submatrix(A, vertices)
        
        # Check mapping
        np.testing.assert_array_equal(mapping, [0, 2])
        
        # Check submatrix shape
        assert submat.shape == (2, 2)
        
        # Expected submatrix (only edges between 0 and 2):
        # Original: A[0,0]=1, A[0,2]=2, A[2,0]=5, A[2,2]=6
        # In submat: [0,0]=1, [0,1]=2, [1,0]=5, [1,1]=6
        expected = np.array([[1, 2], [5, 6]], dtype=np.int32)
        np.testing.assert_array_equal(submat.toarray(), expected)
    
    def test_extract_submatrix_preserves_sparsity(self):
        """Test that extraction preserves sparsity."""
        from redblackgraph.sparse.csgraph import extract_submatrix, get_density
        
        n = 100
        A = sparse_random(n, n, density=0.05, format='csr', dtype=np.int32)
        
        # Extract half the vertices
        vertices = np.arange(0, n, 2, dtype=np.int32)
        submat, _ = extract_submatrix(A, vertices)
        
        # Should still be sparse
        assert get_density(submat) < 0.5
    
    def test_extract_empty_submatrix(self):
        """Test extraction with no internal edges."""
        from redblackgraph.sparse.csgraph import extract_submatrix
        
        # Matrix with edges only between even and odd vertices
        data = [1, 1]
        row = [0, 2]
        col = [1, 3]
        A = csr_matrix((data, (row, col)), shape=(4, 4), dtype=np.int32)
        
        # Extract only even vertices (no edges between them)
        vertices = np.array([0, 2], dtype=np.int32)
        submat, _ = extract_submatrix(A, vertices)
        
        assert submat.nnz == 0


class TestMergeComponentMatrices:
    """Tests for component matrix merging."""
    
    def test_merge_basic(self):
        """Test basic component merging."""
        from redblackgraph.sparse.csgraph import (
            extract_submatrix, merge_component_matrices
        )
        
        # Create matrix with 2 components
        # Component 0: vertices 0, 1
        # Component 1: vertices 2, 3
        dense = np.array([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8],
        ], dtype=np.int32)
        A = csr_matrix(dense)
        
        # Extract components
        comp0_vertices = np.array([0, 1], dtype=np.int32)
        comp1_vertices = np.array([2, 3], dtype=np.int32)
        
        sub0, map0 = extract_submatrix(A, comp0_vertices)
        sub1, map1 = extract_submatrix(A, comp1_vertices)
        
        # Merge back
        merged = merge_component_matrices([(sub0, map0), (sub1, map1)], 4)
        
        np.testing.assert_array_equal(merged.toarray(), dense)
    
    def test_extract_merge_roundtrip(self):
        """Test that extract → merge roundtrip preserves data."""
        from redblackgraph.sparse.csgraph import (
            find_components_sparse, extract_submatrix, 
            merge_component_matrices, get_component_vertices
        )
        
        # Create sparse matrix
        n = 50
        A = sparse_random(n, n, density=0.1, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)
        
        # Find components
        component_labels = find_components_sparse(A)
        comp_vertices = get_component_vertices(component_labels)
        
        # Extract each component
        components = []
        for vertices in comp_vertices.values():
            submat, mapping = extract_submatrix(A, vertices)
            components.append((submat, mapping))
        
        # Merge back
        merged = merge_component_matrices(components, n)
        
        # Should match original
        np.testing.assert_array_equal(merged.toarray(), A.toarray())


class TestGetComponentVertices:
    """Tests for get_component_vertices utility."""
    
    def test_get_component_vertices_basic(self):
        """Test basic component vertex extraction."""
        from redblackgraph.sparse.csgraph import get_component_vertices
        
        labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.uint32)
        
        result = get_component_vertices(labels)
        
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0, 1])
        np.testing.assert_array_equal(result[1], [2, 3, 4])
        np.testing.assert_array_equal(result[2], [5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
