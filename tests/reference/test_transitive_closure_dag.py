"""
Tests for reference implementation of sparse DAG transitive closure.
"""

import pytest
from redblackgraph import RED_ONE, BLACK_ONE
from redblackgraph.reference import transitive_closure_dag, ReferenceCycleError
from redblackgraph.reference.rbg_math import avos_sum, avos_product


class TestTransitiveClosureDag:
    """Tests for reference DAG transitive closure."""
    
    def test_simple_chain(self):
        """Test on simple chain: 0 -> 1 -> 2."""
        A = [
            [1, 2, 0],
            [0, 1, 4],
            [0, 0, 1]
        ]
        
        result = transitive_closure_dag(A)
        W = result.W
        
        # Should have transitive edge 0 -> 2
        # 0 -> 1 (val=2), 1 -> 2 (val=4)
        # avos_product(2, 4) = 8
        assert W[0][2] == 8
    
    def test_known_small_dag(self):
        """Test on known small DAG with expected result."""
        # DAG: 0 -> 1 (val=2), 0 -> 2 (val=4), 1 -> 2 (val=4), 2 -> 3 (val=8)
        A = [
            [1, 2, 4, 0],
            [0, 1, 4, 0],
            [0, 0, 1, 8],
            [0, 0, 0, 1]
        ]
        
        result = transitive_closure_dag(A)
        W = result.W
        
        # Expected closure (calculated using AVOS operations):
        # 0 -> 1: 2 (direct)
        # 0 -> 2: avos_sum(4, avos_product(2, 4)) = avos_sum(4, 8) = 4 (direct wins)
        # 0 -> 3: avos_sum(avos_product(4, 8), avos_product(8, 8)) = avos_sum(32, 64) = 32
        # 1 -> 2: 4 (direct)
        # 1 -> 3: avos_product(4, 8) = 32 (via 1->2->3)
        # 2 -> 3: 8 (direct)
        expected = [
            [1, 2, 4, 32],
            [0, 1, 4, 32],
            [0, 0, 1, 8],
            [0, 0, 0, 1]
        ]
        
        assert W == expected
    
    def test_two_paths_avos_composition(self):
        """Test AVOS composition when two paths lead to same target."""
        # Diamond DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
        A = [
            [1, 2, 4, 0],  # 0 -> 1 (val=2), 0 -> 2 (val=4)
            [0, 1, 0, 8],  # 1 -> 3 (val=8)
            [0, 0, 1, 4],  # 2 -> 3 (val=4)
            [0, 0, 0, 1]
        ]
        
        result = transitive_closure_dag(A)
        W = result.W
        
        # Path 1: 0 -> 1 -> 3: avos_product(2, 8) = 16
        # Path 2: 0 -> 2 -> 3: avos_product(4, 4) = 12
        # Final: avos_sum(16, 12) = 12 (minimum non-zero)
        path1 = avos_product(2, 8)  # 16
        path2 = avos_product(4, 4)  # 12
        expected_0_3 = avos_sum(path1, path2)  # 12
        
        assert W[0][3] == expected_0_3
    
    def test_single_vertex(self):
        """Test with single vertex."""
        A = [[1]]
        result = transitive_closure_dag(A)
        
        assert result.W[0][0] == 1
    
    def test_empty_matrix(self):
        """Test with empty matrix."""
        A = []
        result = transitive_closure_dag(A)
        
        assert result.W == []
    
    def test_isolated_vertices(self):
        """Test with isolated vertices (no edges except self-loops)."""
        n = 5
        A = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        
        result = transitive_closure_dag(A)
        
        # Should just be identity
        expected = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        assert result.W == expected
    
    def test_raises_cycle_error(self):
        """Test that CycleError is raised for cyclic graphs."""
        # Simple cycle: 0 -> 1 -> 0
        A = [
            [1, 2],
            [2, 1]
        ]
        
        with pytest.raises(ReferenceCycleError):
            transitive_closure_dag(A)
    
    def test_parity_identities(self):
        """Test with RED_ONE and BLACK_ONE identity values."""
        # Mix of RED_ONE (-1) and BLACK_ONE (1) on diagonal
        A = [
            [RED_ONE, 2, 0],
            [0, BLACK_ONE, 4],
            [0, 0, RED_ONE]
        ]
        
        result = transitive_closure_dag(A)
        W = result.W
        
        # Diagonal should preserve identity values
        assert W[0][0] == RED_ONE
        assert W[1][1] == BLACK_ONE
        assert W[2][2] == RED_ONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
