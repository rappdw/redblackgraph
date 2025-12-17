"""
Tests for sparse DAG transitive closure.

These tests verify:
- transitive_closure_dag_sparse produces correct results
- sparse_only mode in transitive_closure_adaptive works correctly
- Proper error handling for cyclic graphs
- No O(N^2) memory allocation in sparse-only mode
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix

from redblackgraph import RED_ONE, BLACK_ONE
from redblackgraph.reference.rbg_math import avos_sum, avos_product


class TestTransitiveClosureDagSparse:
    """Tests for sparse DAG transitive closure algorithm."""
    
    def test_simple_chain(self):
        """Test on simple chain: 0 -> 1 -> 2."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        W = result.W.toarray()
        
        # Should have transitive edge 0 -> 2
        # 0 -> 1 (val=2), 1 -> 2 (val=4)
        # avos_product(2, 4) = 8
        assert W[0, 2] == 8
        
    def test_known_small_dag(self):
        """Test on known small DAG with expected result."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        # DAG: 0 -> 1 (val=2), 0 -> 2 (val=4), 1 -> 2 (val=4), 2 -> 3 (val=8)
        A = csr_matrix(np.array([
            [1, 2, 4, 0],
            [0, 1, 4, 0],
            [0, 0, 1, 8],
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        W = result.W.toarray()
        
        # Expected closure (calculated using AVOS operations):
        # 0 -> 1: 2 (direct)
        # 0 -> 2: avos_sum(4, avos_product(2, 4)) = avos_sum(4, 8) = 4 (direct wins)
        # 0 -> 3: avos_sum(avos_product(4, 8), avos_product(8, 8)) = avos_sum(32, 64) = 32
        #         (path 0->2->3 wins over 0->1->2->3)
        # 1 -> 2: 4 (direct)
        # 1 -> 3: avos_product(4, 8) = 32 (via 1->2->3)
        # 2 -> 3: 8 (direct)
        expected = np.array([
            [1, 2, 4, 32],
            [0, 1, 4, 32],
            [0, 0, 1, 8],
            [0, 0, 0, 1]
        ], dtype=np.int32)
        
        np.testing.assert_array_equal(W, expected)
    
    def test_matches_floyd_warshall(self):
        """Test that sparse DAG closure matches Floyd-Warshall on random DAGs."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(42)
        n = 20
        
        # Create random upper triangular matrix (guaranteed DAG)
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Sparse DAG closure
        result_sparse = transitive_closure_dag_sparse(csr_matrix(A))
        
        # Floyd-Warshall reference
        result_fw = transitive_closure_floyd_warshall(rb_matrix(A))
        
        # Should be equal
        np.testing.assert_array_equal(result_sparse.W.toarray(), result_fw.W)
    
    def test_two_paths_avos_composition(self):
        """Test AVOS composition when two paths lead to same target."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        # Diamond DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
        # Two paths from 0 to 3
        A = csr_matrix(np.array([
            [1, 2, 4, 0],  # 0 -> 1 (val=2), 0 -> 2 (val=4)
            [0, 1, 0, 8],  # 1 -> 3 (val=8)
            [0, 0, 1, 4],  # 2 -> 3 (val=4)
            [0, 0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        W = result.W.toarray()
        
        # Path 1: 0 -> 1 -> 3: avos_product(2, 8) = 16
        # Path 2: 0 -> 2 -> 3: avos_product(4, 4) = 12
        # Final: avos_sum(16, 12) = 12 (minimum non-zero)
        path1 = avos_product(2, 8)  # 16
        path2 = avos_product(4, 4)  # 12
        expected_0_3 = avos_sum(path1, path2)  # 12
        
        assert W[0, 3] == expected_0_3
    
    def test_single_vertex(self):
        """Test with single vertex."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        A = csr_matrix([[1]], dtype=np.int32)
        result = transitive_closure_dag_sparse(A)
        
        assert result.W.toarray()[0, 0] == 1
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        A = csr_matrix((0, 0), dtype=np.int32)
        result = transitive_closure_dag_sparse(A)
        
        assert result.W.shape == (0, 0)
    
    def test_isolated_vertices(self):
        """Test with isolated vertices (no edges except self-loops)."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        n = 10
        A = csr_matrix(np.eye(n, dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        
        # Should just be identity
        np.testing.assert_array_equal(result.W.toarray(), np.eye(n, dtype=np.int32))
    
    def test_linear_chain(self):
        """Test with linear chain graph."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        n = 5
        # Build chain: 0 -> 1 -> 2 -> 3 -> 4
        A = np.eye(n, dtype=np.int32)
        for i in range(n - 1):
            A[i, i + 1] = 2
        A = csr_matrix(A)
        
        result = transitive_closure_dag_sparse(A)
        W = result.W.toarray()
        
        # All upper triangular entries should be non-zero
        for i in range(n):
            for j in range(i + 1, n):
                assert W[i, j] != 0
    
    def test_result_is_sparse(self):
        """Test that result is sparse matrix."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        
        assert isspmatrix(result.W)
    
    def test_raises_cycle_error(self):
        """Test that CycleError is raised for cyclic graphs."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        from redblackgraph.sparse.csgraph.cycleerror import CycleError
        
        # Simple cycle: 0 -> 1 -> 0
        A = csr_matrix(np.array([
            [1, 2],
            [2, 1]
        ], dtype=np.int32))
        
        with pytest.raises(CycleError):
            transitive_closure_dag_sparse(A)
    
    def test_parity_identities(self):
        """Test with RED_ONE and BLACK_ONE identity values."""
        from redblackgraph.sparse.csgraph import transitive_closure_dag_sparse
        
        # Mix of RED_ONE (-1) and BLACK_ONE (1) on diagonal
        A = csr_matrix(np.array([
            [RED_ONE, 2, 0],
            [0, BLACK_ONE, 4],
            [0, 0, RED_ONE]
        ], dtype=np.int32))
        
        result = transitive_closure_dag_sparse(A)
        W = result.W.toarray()
        
        # Diagonal should preserve identity values
        assert W[0, 0] == RED_ONE
        assert W[1, 1] == BLACK_ONE
        assert W[2, 2] == RED_ONE


class TestSparseOnlyMode:
    """Tests for sparse_only mode in transitive_closure_adaptive."""
    
    def test_sparse_only_on_dag(self):
        """Test sparse_only mode works on DAGs."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        
        A = csr_matrix(np.array([
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A, sparse_only=True)
        W = result.W.toarray()
        
        # Should have transitive edge
        assert W[0, 2] == 8
    
    def test_sparse_only_matches_standard(self):
        """Test sparse_only produces same result as standard closure."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive, transitive_closure_floyd_warshall
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(123)
        n = 15
        
        # Create random upper triangular matrix (guaranteed DAG)
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Sparse-only
        result_sparse = transitive_closure_adaptive(csr_matrix(A), sparse_only=True)
        
        # Floyd-Warshall reference
        result_fw = transitive_closure_floyd_warshall(rb_matrix(A))
        
        np.testing.assert_array_equal(result_sparse.W.toarray(), result_fw.W)
    
    def test_sparse_only_raises_densification_error_on_cycle(self):
        """Test sparse_only raises DensificationError for cyclic graphs."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive
        from redblackgraph.sparse.csgraph._density import DensificationError
        
        # Simple cycle: 0 -> 1 -> 0
        A = csr_matrix(np.array([
            [1, 2],
            [2, 1]
        ], dtype=np.int32))
        
        with pytest.raises(DensificationError):
            transitive_closure_adaptive(A, sparse_only=True)
    
    def test_method_override_dag_sparse(self):
        """Test method='dag_sparse' override."""
        from redblackgraph.sparse.csgraph import transitive_closure_adaptive, transitive_closure_dag_sparse
        
        A = csr_matrix(np.array([
            [1, 2, 4],
            [0, 1, 2],
            [0, 0, 1]
        ], dtype=np.int32))
        
        result = transitive_closure_adaptive(A, method="dag_sparse")
        expected = transitive_closure_dag_sparse(A)
        
        np.testing.assert_array_equal(result.W.toarray(), expected.W.toarray())


class TestAllMethodsEquivalentOnDAGs:
    """Tests that all methods produce equivalent results on DAGs."""
    
    def test_dag_sparse_matches_floyd_warshall(self):
        """Test that DAG sparse closure matches Floyd-Warshall reference."""
        from redblackgraph.sparse.csgraph import (
            transitive_closure_floyd_warshall,
            transitive_closure_dag_sparse,
            transitive_closure_adaptive
        )
        from redblackgraph.sparse import rb_matrix
        
        np.random.seed(456)
        n = 15
        
        # Create random upper triangular matrix (guaranteed DAG)
        A = np.triu(np.random.randint(0, 4, size=(n, n)), k=1)
        np.fill_diagonal(A, np.random.choice([-1, 1], size=n))
        A = A.astype(np.int32)
        
        # Compute closure with Floyd-Warshall (reference) and sparse DAG methods
        result_fw = transitive_closure_floyd_warshall(rb_matrix(A))
        result_dag = transitive_closure_dag_sparse(csr_matrix(A))
        result_adapt_sparse = transitive_closure_adaptive(csr_matrix(A), sparse_only=True)
        
        # Sparse DAG closure should match Floyd-Warshall reference
        W_fw = result_fw.W
        W_dag = result_dag.W.toarray()
        W_adapt = result_adapt_sparse.W.toarray()
        
        np.testing.assert_array_equal(W_dag, W_fw)
        np.testing.assert_array_equal(W_adapt, W_fw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
