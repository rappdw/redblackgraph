"""
Tests for sparse topological sort.

These tests verify:
- Correctness: resulting permutation produces upper triangular matrix
- Cycle detection: raises CycleError for graphs with cycles
- Complexity: O(V+E) scaling for sparse graphs
- Multi-component handling
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestTopologicalSortBasic:
    """Basic correctness tests for topological_sort."""
    
    def test_simple_dag(self):
        """Test topological sort on a simple DAG."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        # Simple DAG: 0 -> 1 -> 2
        # A[i,j] != 0 means edge i -> j
        # Upper triangular = already in topological order
        data = [1, 2, 1, 4, 1]  # diagonal + edges
        row = [0, 0, 1, 1, 2]
        col = [0, 1, 1, 2, 2]
        A = csr_matrix((data, (row, col)), shape=(3, 3), dtype=np.int32)
        
        p = topological_sort(A)
        
        # Check p is a valid permutation
        assert len(p) == 3
        assert set(p) == {0, 1, 2}
    
    def test_already_upper_triangular(self):
        """Test that an already upper triangular matrix returns identity-like permutation."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Upper triangular DAG
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        assert is_upper_triangular(A)
        
        p = topological_sort(A)
        B = permute_sparse(A, p)
        
        # Result should still be upper triangular
        assert is_upper_triangular(B)
    
    def test_lower_triangular_reversed(self):
        """Test that lower triangular matrix gets properly reversed."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Lower triangular matrix (edges go from higher to lower index)
        # This represents edges: 1->0, 2->0, 2->1
        A = csr_matrix(np.array([
            [1, 0, 0],
            [2, 1, 0],
            [4, 2, 1]
        ], dtype=np.int32))
        
        p = topological_sort(A)
        B = permute_sparse(A, p, assume_upper_triangular=True)
        
        # After reordering, should be upper triangular
        assert is_upper_triangular(B)
    
    def test_upper_triangular_property(self):
        """Verify that result is always upper triangular after permutation."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Random DAG - create upper triangular first, then permute randomly
        np.random.seed(42)
        n = 50
        
        # Create random upper triangular matrix
        A_dense = np.triu(np.random.randint(0, 5, size=(n, n)), k=0)
        np.fill_diagonal(A_dense, np.random.choice([-1, 1], size=n))  # RBG diagonal
        A = csr_matrix(A_dense.astype(np.int32))
        
        # Randomly permute it
        rand_perm = np.random.permutation(n).astype(np.int32)
        A_permuted = permute_sparse(A, rand_perm)
        
        # Now topological sort should recover upper triangular property
        p = topological_sort(A_permuted)
        B = permute_sparse(A_permuted, p, assume_upper_triangular=True)
        
        assert is_upper_triangular(B), "Result should be upper triangular"


class TestTopologicalSortCycleDetection:
    """Tests for cycle detection in topological_sort."""
    
    def test_self_loop_skipped(self):
        """Test that self-loops (diagonal) don't cause cycle detection."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        # DAG with self-loops on diagonal
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        # Should not raise - diagonal entries are not cycles
        p = topological_sort(A)
        assert len(p) == 3
    
    def test_simple_cycle_detected(self):
        """Test that a simple cycle is detected."""
        from redblackgraph.sparse.csgraph import topological_sort
        from redblackgraph.sparse.csgraph.cycleerror import CycleError
        
        # Cycle: 0 -> 1 -> 2 -> 0
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [8, 0, 1]  # 2 -> 0 creates cycle
        ], dtype=np.int32))
        
        with pytest.raises(CycleError):
            topological_sort(A)
    
    def test_two_node_cycle(self):
        """Test that a two-node cycle is detected."""
        from redblackgraph.sparse.csgraph import topological_sort
        from redblackgraph.sparse.csgraph.cycleerror import CycleError
        
        # Cycle: 0 <-> 1
        A = csr_matrix(np.array([
            [1, 2],
            [4, 1]  # 1 -> 0 creates cycle
        ], dtype=np.int32))
        
        with pytest.raises(CycleError):
            topological_sort(A)


class TestTopologicalSortMultiComponent:
    """Tests for graphs with multiple disconnected components."""
    
    def test_two_components(self):
        """Test topological sort with two disconnected components."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Two disconnected DAGs
        # Component 1: 0 -> 1
        # Component 2: 2 -> 3
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        p = topological_sort(A)
        B = permute_sparse(A, p, assume_upper_triangular=True)
        
        assert is_upper_triangular(B)
    
    def test_many_singleton_components(self):
        """Test with many single-vertex components."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        # Diagonal matrix - each vertex is its own component
        n = 100
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        p = topological_sort(A)
        
        assert len(p) == n
        assert set(p) == set(range(n))
    
    def test_mixed_components(self):
        """Test with mixed component sizes."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        # Build a matrix with:
        # - One 3-vertex component (0->1->2)
        # - Two 1-vertex components (3, 4)
        # - One 2-vertex component (5->6)
        n = 7
        A = csr_matrix(np.array([
            [1, 2, 0, 0, 0, 0, 0],
            [0, 1, 4, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.int32))
        
        p = topological_sort(A)
        B = permute_sparse(A, p, assume_upper_triangular=True)
        
        assert is_upper_triangular(B)


class TestTopologicalSortSparse:
    """Tests specifically for sparse matrix handling and performance."""
    
    def test_sparse_csr_input(self):
        """Test that CSR input is handled efficiently."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        n = 1000
        # Create sparse upper triangular
        A = sparse_random(n, n, density=0.01, format='csr', dtype=np.int32)
        A.data[:] = np.random.randint(1, 100, size=A.nnz)
        # Make it upper triangular to ensure DAG
        A = csr_matrix(np.triu(A.toarray()))
        
        p = topological_sort(A)
        assert len(p) == n
    
    def test_sparse_csc_input(self):
        """Test that CSC input is converted and works."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        # Convert to CSC
        A_csc = A.tocsc()
        
        p = topological_sort(A_csc)
        assert len(p) == 3
    
    def test_dense_input(self):
        """Test that dense input is handled."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        A = np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32)
        
        p = topological_sort(A)
        assert len(p) == 3
    
    def test_empty_matrix(self):
        """Test empty matrix handling."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        A = csr_matrix((0, 0), dtype=np.int32)
        p = topological_sort(A)
        assert len(p) == 0
    
    def test_single_vertex(self):
        """Test single vertex graph."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        A = csr_matrix([[1]], dtype=np.int32)
        p = topological_sort(A)
        np.testing.assert_array_equal(p, [0])


class TestTopologicalSortComplexity:
    """Tests for O(V+E) complexity."""
    
    def test_complexity_fixed_density(self):
        """Test that time scales with V+E, not V²."""
        from redblackgraph.sparse.csgraph import topological_sort
        
        np.random.seed(42)
        times = []
        sizes = [1000, 5000, 10000]
        
        for n in sizes:
            # Fixed density -> E scales with n²
            # But we want O(V+E), so use fixed nnz
            nnz_target = 10000
            density = min(nnz_target / (n * n), 0.5)
            
            # Create random upper triangular (DAG)
            A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
            A.data[:] = np.random.randint(1, 100, size=A.nnz)
            A = csr_matrix(np.triu(A.toarray()))
            
            start = time.perf_counter()
            p = topological_sort(A)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # With fixed nnz, time should not grow much with n
        # Allow 20x tolerance for system noise
        assert times[-1] < times[0] * 20, \
            f"Time growing too fast with n: {list(zip(sizes, times))}"
    
    def test_large_sparse_graph(self):
        """Test on a large sparse graph (100K vertices)."""
        from redblackgraph.sparse.csgraph import topological_sort, is_upper_triangular
        from redblackgraph.sparse.csgraph import permute_sparse
        
        np.random.seed(42)
        n = 100000
        
        # Create a very sparse DAG - linear chain plus some random edges
        # This ensures we have a valid DAG
        row = list(range(n - 1))  # 0, 1, ..., n-2
        col = list(range(1, n))   # 1, 2, ..., n-1
        data = [1] * (n - 1)
        
        # Add some random forward edges (j > i to maintain DAG)
        np.random.seed(42)
        for _ in range(min(5000, n)):
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 1, n)
            row.append(i)
            col.append(j)
            data.append(np.random.randint(1, 100))
        
        A = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
        
        start = time.perf_counter()
        p = topological_sort(A)
        elapsed = time.perf_counter() - start
        
        print(f"Topological sort on {n} vertices, {A.nnz} edges: {elapsed:.3f}s")
        
        # Should complete in reasonable time (<5 seconds on modern hardware)
        assert elapsed < 5.0, f"Topological sort too slow: {elapsed}s"
        
        # Verify correctness on a sample
        B = permute_sparse(A, p, assume_upper_triangular=True)
        assert is_upper_triangular(B)


class TestTopologicalOrdering:
    """Tests for the topological_ordering wrapper function."""
    
    def test_ordering_returns_correct_type(self):
        """Test that topological_ordering returns an Ordering object."""
        from redblackgraph.sparse.csgraph import topological_ordering
        from redblackgraph.types.ordering import Ordering
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = topological_ordering(A)
        
        assert isinstance(result, Ordering)
        assert hasattr(result, 'A')
        assert hasattr(result, 'label_permutation')
        assert hasattr(result, 'components')
    
    def test_ordering_matrix_is_upper_triangular(self):
        """Test that the resulting matrix is upper triangular."""
        from redblackgraph.sparse.csgraph import topological_ordering, is_upper_triangular
        
        # Lower triangular input
        A = csr_matrix(np.array([
            [1, 0, 0],
            [2, 1, 0],
            [4, 2, 1]
        ], dtype=np.int32))
        
        result = topological_ordering(A)
        
        assert is_upper_triangular(result.A)
    
    def test_ordering_preserves_structure(self):
        """Test that ordering preserves the graph structure."""
        from redblackgraph.sparse.csgraph import topological_ordering
        from scipy.sparse import isspmatrix
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = topological_ordering(A)
        
        # Result should also be sparse
        assert isspmatrix(result.A)
        
        # Should have same or fewer nnz (upper triangular filter)
        assert result.A.nnz <= A.nnz
    
    def test_ordering_components_dict(self):
        """Test that components dictionary is populated."""
        from redblackgraph.sparse.csgraph import topological_ordering
        
        # Two disconnected components
        A = csr_matrix(np.array([
            [1, 2, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = topological_ordering(A)
        
        # Should have component info
        assert isinstance(result.components, dict)
        assert len(result.components) > 0


class TestIsUpperTriangular:
    """Tests for the is_upper_triangular helper function."""
    
    def test_upper_triangular_true(self):
        """Test detection of upper triangular matrix."""
        from redblackgraph.sparse.csgraph import is_upper_triangular
        
        A = csr_matrix(np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ], dtype=np.int32))
        
        assert is_upper_triangular(A)
        assert not is_upper_triangular(A, strict=True)  # Has diagonal
    
    def test_lower_triangular_false(self):
        """Test detection of lower triangular matrix."""
        from redblackgraph.sparse.csgraph import is_upper_triangular
        
        A = csr_matrix(np.array([
            [1, 0, 0],
            [2, 3, 0],
            [4, 5, 6]
        ], dtype=np.int32))
        
        assert not is_upper_triangular(A)
    
    def test_strictly_upper_triangular(self):
        """Test strict upper triangular detection."""
        from redblackgraph.sparse.csgraph import is_upper_triangular
        
        # Strictly upper triangular (no diagonal)
        A = csr_matrix(np.array([
            [0, 2, 3],
            [0, 0, 5],
            [0, 0, 0]
        ], dtype=np.int32))
        
        assert is_upper_triangular(A, strict=True)
        assert is_upper_triangular(A, strict=False)
    
    def test_dense_input(self):
        """Test is_upper_triangular with dense input."""
        from redblackgraph.sparse.csgraph import is_upper_triangular
        
        A = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ], dtype=np.int32)
        
        assert is_upper_triangular(A)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
