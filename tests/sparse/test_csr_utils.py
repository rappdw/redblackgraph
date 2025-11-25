"""
Tests for CSR iteration primitives and sparse utilities.

These tests verify that the sparse infrastructure (Phase 0) works correctly
and maintains O(nnz) complexity instead of O(n²).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random
import time


class TestCSRIteration:
    """Test CSR iteration primitives indirectly through higher-level functions."""
    
    def test_sparse_iteration_visits_only_nonzeros(self):
        """Verify iteration only visits non-zero entries."""
        # Create sparse matrix with known structure
        n = 100
        data = [1, 2, 3, 4, 5]
        row = [0, 0, 1, 2, 2]
        col = [0, 5, 10, 20, 25]
        A = csr_matrix((data, (row, col)), shape=(n, n))
        
        # Access through indptr/indices should only touch nnz entries
        assert A.nnz == 5
        
        visited_count = 0
        for i in range(n):
            for j_idx in range(A.indptr[i], A.indptr[i + 1]):
                visited_count += 1
        
        assert visited_count == 5, "Should only visit non-zero entries"
    
    def test_csr_row_access_pattern(self):
        """Test CSR row access follows expected pattern."""
        # Row 0: columns [0, 2]
        # Row 1: columns [1]
        # Row 2: empty
        # Row 3: columns [0, 1, 2, 3]
        indptr = np.array([0, 2, 3, 3, 7])
        indices = np.array([0, 2, 1, 0, 1, 2, 3])
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        
        A = csr_matrix((data, indices, indptr), shape=(4, 4))
        
        # Verify row access
        assert list(A.indices[A.indptr[0]:A.indptr[1]]) == [0, 2]
        assert list(A.indices[A.indptr[1]:A.indptr[2]]) == [1]
        assert list(A.indices[A.indptr[2]:A.indptr[3]]) == []
        assert list(A.indices[A.indptr[3]:A.indptr[4]]) == [0, 1, 2, 3]
    
    def test_iteration_complexity_scaling(self):
        """Verify O(nnz) not O(n²) scaling for sparse iteration."""
        # Compare sparse iteration vs dense iteration
        # Sparse should be dramatically faster for sparse matrices
        n = 5000
        density = 0.001  # 0.1% density
        A = sparse_random(n, n, density=density, format='csr', dtype=np.int32)
        A_dense = A.toarray()
        
        # Time sparse CSR iteration (O(nnz))
        start = time.perf_counter()
        sparse_count = 0
        for i in range(n):
            for j_idx in range(A.indptr[i], A.indptr[i + 1]):
                sparse_count += 1
        sparse_time = time.perf_counter() - start
        
        # Time would-be dense iteration (O(n²)) - just measure loop overhead
        start = time.perf_counter()
        dense_loop_count = 0
        for i in range(n):
            for j in range(n):
                dense_loop_count += 1
                if dense_loop_count > 100000:  # Early exit to not wait forever
                    break
            if dense_loop_count > 100000:
                break
        dense_partial_time = time.perf_counter() - start
        
        # Extrapolate dense time for full iteration
        full_dense_iterations = n * n
        estimated_dense_time = dense_partial_time * (full_dense_iterations / dense_loop_count)
        
        # Sparse should be at least 10x faster than dense for this density
        assert sparse_time < estimated_dense_time / 10, \
            f"Sparse ({sparse_time:.4f}s) should be much faster than dense ({estimated_dense_time:.4f}s)"


class TestSparseFormat:
    """Tests for sparse format conversion utilities."""
    
    def test_ensure_csr_from_dense(self):
        """Test CSR conversion from dense array."""
        from redblackgraph.sparse.csgraph import ensure_csr
        
        dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        sparse = ensure_csr(dense)
        
        assert sparse.format == 'csr'
        assert sparse.nnz == 5
        np.testing.assert_array_equal(sparse.toarray(), dense)
    
    def test_ensure_csr_passthrough(self):
        """Test that CSR input is passed through without copy."""
        from redblackgraph.sparse.csgraph import ensure_csr
        
        A = csr_matrix(np.eye(5, dtype=np.int32))
        B = ensure_csr(A, copy=False)
        
        assert A is B, "Should return same object without copy"
    
    def test_ensure_csc_from_csr(self):
        """Test CSC conversion from CSR."""
        from redblackgraph.sparse.csgraph import ensure_csc
        
        A_csr = csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32))
        A_csc = ensure_csc(A_csr)
        
        assert A_csc.format == 'csc'
        np.testing.assert_array_equal(A_csr.toarray(), A_csc.toarray())
    
    def test_get_density(self):
        """Test density calculation."""
        from redblackgraph.sparse.csgraph import get_density
        
        # 4 non-zeros in 4x4 matrix = 25% density
        A = csr_matrix(np.eye(4, dtype=np.int32))
        assert get_density(A) == 0.25
        
        # Empty matrix
        B = csr_matrix((10, 10), dtype=np.int32)
        assert get_density(B) == 0.0
        
        # Full matrix
        C = np.ones((3, 3), dtype=np.int32)
        assert get_density(C) == 1.0
    
    def test_csr_csc_pair(self):
        """Test getting both CSR and CSC representations."""
        from redblackgraph.sparse.csgraph import csr_csc_pair
        
        dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int32)
        A_csr, A_csc = csr_csc_pair(dense)
        
        assert A_csr.format == 'csr'
        assert A_csc.format == 'csc'
        np.testing.assert_array_equal(A_csr.toarray(), A_csc.toarray())
    
    def test_get_format(self):
        """Test format detection."""
        from redblackgraph.sparse.csgraph import get_format
        
        assert get_format(csr_matrix((5, 5))) == 'csr'
        assert get_format(np.zeros((5, 5))) == 'dense'


class TestDensityMonitoring:
    """Tests for density monitoring utilities."""
    
    def test_density_monitor_basic(self):
        """Test basic density monitoring."""
        from redblackgraph.sparse.csgraph import DensityMonitor
        
        monitor = DensityMonitor(warn_threshold=0.3, error_threshold=0.8)
        
        # Low density - should pass
        A = csr_matrix(np.eye(10, dtype=np.int32))  # 10% density
        density = monitor.check(A, "test_op", warn_on_threshold=False)
        assert density == 0.1
        assert len(monitor.history) == 1
    
    def test_density_monitor_warning(self):
        """Test density warning threshold."""
        from redblackgraph.sparse.csgraph import DensityMonitor, DensificationWarning
        
        monitor = DensityMonitor(warn_threshold=0.05, error_threshold=0.8)
        
        # Medium density - should warn
        A = csr_matrix(np.eye(10, dtype=np.int32))  # 10% > 5% threshold
        
        with pytest.warns(DensificationWarning):
            monitor.check(A, "test_op")
    
    def test_density_monitor_error(self):
        """Test density error threshold."""
        from redblackgraph.sparse.csgraph import DensityMonitor, DensificationError
        
        monitor = DensityMonitor(warn_threshold=0.05, error_threshold=0.3)
        
        # High density - should error
        A = np.ones((5, 5), dtype=np.int32)  # 100% > 30% threshold
        
        with pytest.raises(DensificationError) as exc_info:
            monitor.check(A, "dense_op")
        
        assert exc_info.value.density == 1.0
        assert exc_info.value.threshold == 0.3
    
    def test_density_history(self):
        """Test density history tracking."""
        from redblackgraph.sparse.csgraph import DensityMonitor
        
        monitor = DensityMonitor(warn_threshold=1.0, error_threshold=1.0)
        
        monitor.check(csr_matrix(np.eye(10, dtype=np.int32)), "op1")
        monitor.check(csr_matrix(np.eye(5, dtype=np.int32)), "op2")
        monitor.check(csr_matrix((10, 10), dtype=np.int32), "op3")
        
        assert len(monitor.history) == 3
        assert monitor.history[0].operation == "op1"
        assert monitor.history[0].density == 0.1
        assert monitor.history[1].density == 0.2
        assert monitor.history[2].density == 0.0
    
    def test_assert_sparse(self):
        """Test assert_sparse utility."""
        from redblackgraph.sparse.csgraph import assert_sparse, DensificationError
        
        sparse = csr_matrix(np.eye(100, dtype=np.int32))  # 1% density
        dense = np.ones((10, 10), dtype=np.int32)  # 100% density
        
        # Should pass
        assert_sparse(sparse, max_density=0.1)
        
        # Should fail
        with pytest.raises(DensificationError):
            assert_sparse(dense, max_density=0.5)


class TestTransposeUtilities:
    """Tests for transpose utilities."""
    
    def test_csr_csc_transpose_equivalence(self):
        """Test that CSR transpose equals CSC."""
        from redblackgraph.sparse.csgraph import ensure_csr, ensure_csc
        
        A = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=np.int32)
        A_csr = ensure_csr(A)
        A_csc = ensure_csc(A)
        
        # A^T in CSR == A in CSC (structurally)
        A_T_csr = A_csr.T.tocsr()
        
        np.testing.assert_array_equal(A_T_csr.toarray(), A.T)
        np.testing.assert_array_equal(A_csc.toarray(), A)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
