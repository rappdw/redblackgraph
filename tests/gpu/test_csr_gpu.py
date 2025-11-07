"""
Tests for production CSR GPU matrix data structure.

Tests the new CSRMatrixGPU class with raw int32 buffers.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    from redblackgraph.gpu.csr_gpu import CSRMatrixGPU, validate_triangular_mask
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")


@pytest.fixture
def simple_csr_data():
    """Simple 3x3 upper triangular matrix."""
    # Matrix:
    # [2  3  0]
    # [0  4  5]
    # [0  0  6]
    data = cp.array([2, 3, 4, 5, 6], dtype=cp.int32)
    indices = cp.array([0, 1, 1, 2, 2], dtype=cp.int32)
    indptr = cp.array([0, 2, 4, 5], dtype=cp.int32)
    shape = (3, 3)
    return data, indices, indptr, shape


@pytest.fixture
def non_triangular_csr():
    """Non-triangular 3x3 matrix."""
    # Matrix:
    # [2  3  0]
    # [4  5  6]  <- row 1 has col 0, violates triangular
    # [0  0  7]
    data = cp.array([2, 3, 4, 5, 6, 7], dtype=cp.int32)
    indices = cp.array([0, 1, 0, 1, 2, 2], dtype=cp.int32)
    indptr = cp.array([0, 2, 5, 6], dtype=cp.int32)
    shape = (3, 3)
    return data, indices, indptr, shape


class TestCSRMatrixGPU:
    """Test CSRMatrixGPU data structure."""
    
    def test_create_valid(self, simple_csr_data):
        """Test creating a valid CSR matrix."""
        data, indices, indptr, shape = simple_csr_data
        
        A = CSRMatrixGPU(data, indices, indptr, shape, triangular=False)
        
        assert A.shape == shape
        assert A.nnz == 5
        assert A.dtype == cp.int32
        assert not A.triangular
    
    def test_create_triangular(self, simple_csr_data):
        """Test creating an upper triangular CSR matrix."""
        data, indices, indptr, shape = simple_csr_data
        
        A = CSRMatrixGPU(data, indices, indptr, shape, triangular=True)
        
        assert A.shape == shape
        assert A.triangular
    
    def test_reject_non_triangular(self, non_triangular_csr):
        """Test that non-triangular matrices are rejected when triangular=True."""
        data, indices, indptr, shape = non_triangular_csr
        
        with pytest.raises(ValueError, match="Not upper triangular"):
            CSRMatrixGPU(data, indices, indptr, shape, triangular=True)
    
    def test_validate_indptr_bounds(self):
        """Test indptr validation."""
        data = cp.array([1, 2, 3], dtype=cp.int32)
        indices = cp.array([0, 1, 2], dtype=cp.int32)
        indptr = cp.array([0, 1, 2, 4], dtype=cp.int32)  # Wrong: last is 4, not 3
        
        with pytest.raises(ValueError, match="indptr"):
            CSRMatrixGPU(data, indices, indptr, (3, 3))
    
    def test_validate_column_bounds(self):
        """Test column index bounds validation."""
        data = cp.array([1, 2], dtype=cp.int32)
        indices = cp.array([0, 5], dtype=cp.int32)  # Wrong: 5 >= 3 (cols)
        indptr = cp.array([0, 1, 2], dtype=cp.int32)
        
        with pytest.raises(ValueError, match="Column indices out of bounds"):
            CSRMatrixGPU(data, indices, indptr, (2, 3))
    
    def test_memory_usage(self, simple_csr_data):
        """Test memory usage reporting."""
        data, indices, indptr, shape = simple_csr_data
        A = CSRMatrixGPU(data, indices, indptr, shape)
        
        usage = A.memory_usage()
        
        assert 'data' in usage
        assert 'indices' in usage
        assert 'indptr' in usage
        assert 'total' in usage
        assert 'total_mb' in usage
        assert usage['total'] == usage['data'] + usage['indices'] + usage['indptr']
    
    def test_repr(self, simple_csr_data):
        """Test string representation."""
        data, indices, indptr, shape = simple_csr_data
        A = CSRMatrixGPU(data, indices, indptr, shape, triangular=True)
        
        repr_str = repr(A)
        
        assert '3x3' in repr_str
        assert 'nnz=5' in repr_str
        assert 'triangular' in repr_str


class TestCPUGPUTransfer:
    """Test CPU ↔ GPU transfer."""
    
    def test_from_cpu_scipy_csr(self):
        """Test creating GPU matrix from scipy CSR."""
        import scipy.sparse as sp
        
        # Create scipy sparse matrix
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        # Transfer to GPU
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu)
        
        assert A_gpu.shape == (3, 3)
        assert A_gpu.nnz == 5
        assert A_gpu.dtype == cp.int32
    
    def test_from_cpu_auto_detect_triangular(self):
        """Test automatic triangular detection."""
        import scipy.sparse as sp
        
        # Upper triangular matrix
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        # Should auto-detect as triangular
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu)
        assert A_gpu.triangular
    
    def test_from_cpu_force_triangular(self):
        """Test forcing triangular flag."""
        import scipy.sparse as sp
        
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        # Force triangular=True
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        assert A_gpu.triangular
    
    def test_to_cpu_round_trip(self):
        """Test GPU → CPU transfer preserves data."""
        import scipy.sparse as sp
        
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu_orig = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        # Round trip
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu_orig)
        A_cpu_back = A_gpu.to_cpu()
        
        # Compare
        assert A_cpu_back.shape == A_cpu_orig.shape
        assert A_cpu_back.nnz == A_cpu_orig.nnz
        assert np.array_equal(A_cpu_back.data, A_cpu_orig.data)
        assert np.array_equal(A_cpu_back.indices, A_cpu_orig.indices)
        assert np.array_equal(A_cpu_back.indptr, A_cpu_orig.indptr)


class TestTriangularMask:
    """Test triangular mask function."""
    
    def test_upper_triangular_mask(self):
        """Test upper triangular mask: j >= i."""
        assert validate_triangular_mask(0, 0) is True
        assert validate_triangular_mask(0, 1) is True
        assert validate_triangular_mask(1, 0) is False
        assert validate_triangular_mask(1, 1) is True
        assert validate_triangular_mask(2, 5) is True
        assert validate_triangular_mask(5, 2) is False


class TestEmptyMatrix:
    """Test edge cases with empty matrices."""
    
    def test_empty_matrix(self):
        """Test creating an empty CSR matrix."""
        data = cp.array([], dtype=cp.int32)
        indices = cp.array([], dtype=cp.int32)
        indptr = cp.array([0, 0, 0], dtype=cp.int32)
        
        A = CSRMatrixGPU(data, indices, indptr, (2, 2))
        
        assert A.shape == (2, 2)
        assert A.nnz == 0
    
    def test_single_element(self):
        """Test 1x1 matrix with single element."""
        data = cp.array([42], dtype=cp.int32)
        indices = cp.array([0], dtype=cp.int32)
        indptr = cp.array([0, 1], dtype=cp.int32)
        
        A = CSRMatrixGPU(data, indices, indptr, (1, 1), triangular=True)
        
        assert A.shape == (1, 1)
        assert A.nnz == 1
        assert A.triangular
