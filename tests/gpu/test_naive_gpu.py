"""
Tests for naive GPU implementation.

These tests validate basic functionality and help understand:
1. CuPy integration
2. GPU memory management
3. Data transfer patterns
4. Basic correctness

Run with: pytest tests/gpu/test_naive_gpu.py -v
Requires: CuPy installed and CUDA-capable GPU available
"""

import pytest
import numpy as np

# Try to import GPU module
try:
    import cupy as cp
    from redblackgraph.gpu import rb_matrix_gpu, avos_sum_gpu, avos_product_gpu
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import CPU reference implementations for validation
from redblackgraph.reference.rbg_math import avos_sum, avos_product


# Skip all tests if CuPy not available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available - GPU tests skipped"),
]


class TestAVOSOperationsGPU:
    """Test basic AVOS operations on GPU."""
    
    def test_avos_sum_basic(self):
        """Test avos_sum on GPU matches CPU reference."""
        test_cases = [
            (0, 0, 0),
            (0, 5, 5),
            (5, 0, 5),
            (3, 5, 3),
            (5, 3, 3),
            (7, 7, 7),
        ]
        
        for x, y, expected in test_cases:
            # GPU version
            x_gpu = cp.array([x], dtype=cp.int32)
            y_gpu = cp.array([y], dtype=cp.int32)
            result_gpu = avos_sum_gpu(x_gpu, y_gpu)
            
            # Transfer back to CPU for comparison
            result = result_gpu.get()[0]
            
            # CPU reference
            expected_cpu = avos_sum(x, y)
            
            assert result == expected_cpu, f"avos_sum({x}, {y}): got {result}, expected {expected_cpu}"
    
    def test_avos_product_basic(self):
        """Test avos_product on GPU matches CPU reference."""
        test_cases = [
            # (x, y, expected)
            (0, 5, 0),
            (5, 0, 0),
            (2, 3, 11),  # Normal case
            (3, 2, 6),   # Normal case
            # Identity cases
            (-1, -1, -1),  # RED_ONE ⊗ RED_ONE
            (1, 1, 1),     # BLACK_ONE ⊗ BLACK_ONE
            (-1, 1, 0),    # RED_ONE ⊗ BLACK_ONE
            (1, -1, 0),    # BLACK_ONE ⊗ RED_ONE
            # Parity filters
            (2, -1, 2),    # Even ⊗ RED_ONE = even
            (3, -1, 0),    # Odd ⊗ RED_ONE = 0
            (2, 1, 0),     # Even ⊗ BLACK_ONE = 0
            (3, 1, 3),     # Odd ⊗ BLACK_ONE = odd
        ]
        
        for x, y, expected in test_cases:
            # GPU version
            x_gpu = cp.array([x], dtype=cp.int32)
            y_gpu = cp.array([y], dtype=cp.int32)
            result_gpu = avos_product_gpu(x_gpu, y_gpu)
            
            # Transfer back to CPU
            result = result_gpu.get()[0]
            
            # CPU reference
            expected_cpu = avos_product(x, y)
            
            assert result == expected_cpu, f"avos_product({x}, {y}): got {result}, expected {expected_cpu}"
    
    def test_avos_operations_vectorized(self):
        """Test vectorized operations on arrays."""
        # Create arrays on CPU
        x_cpu = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        y_cpu = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
        
        # Transfer to GPU
        x_gpu = cp.asarray(x_cpu)
        y_gpu = cp.asarray(y_cpu)
        
        # Compute on GPU
        sum_gpu = avos_sum_gpu(x_gpu, y_gpu)
        
        # Transfer back
        sum_cpu = sum_gpu.get()
        
        # Validate element-wise
        for i in range(len(x_cpu)):
            expected = avos_sum(x_cpu[i], y_cpu[i])
            assert sum_cpu[i] == expected


class TestRBMatrixGPU:
    """Test rb_matrix_gpu basic functionality."""
    
    def test_create_empty(self):
        """Test creating an empty GPU matrix."""
        M = rb_matrix_gpu(None, shape=(10, 10))
        assert M.shape == (10, 10)
        assert M.nnz == 0
    
    def test_create_from_cpu(self):
        """Test creating GPU matrix from CPU sparse matrix."""
        from scipy import sparse as sp_sparse
        
        # Create small CPU matrix
        data = np.array([2, 3, 4], dtype=np.int32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3], dtype=np.int32)
        cpu_matrix = sp_sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        
        # Transfer to GPU
        gpu_matrix = rb_matrix_gpu.from_cpu(cpu_matrix)
        
        assert gpu_matrix.shape == (3, 3)
        assert gpu_matrix.nnz == 3
    
    def test_to_cpu(self):
        """Test transferring GPU matrix back to CPU."""
        from scipy import sparse as sp_sparse
        
        # Create CPU matrix
        data = np.array([2, 3, 4], dtype=np.int32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3], dtype=np.int32)
        cpu_matrix_orig = sp_sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        
        # GPU round trip
        gpu_matrix = rb_matrix_gpu.from_cpu(cpu_matrix_orig)
        cpu_matrix_result = gpu_matrix.to_cpu()
        
        # Validate
        np.testing.assert_array_equal(cpu_matrix_orig.data, cpu_matrix_result.data)
        np.testing.assert_array_equal(cpu_matrix_orig.indices, cpu_matrix_result.indices)
        np.testing.assert_array_equal(cpu_matrix_orig.indptr, cpu_matrix_result.indptr)
    
    @pytest.mark.slow
    def test_matmul_tiny(self):
        """
        Test matrix multiplication on GPU (very small matrix only).
        
        WARNING: This uses the naive O(n³) implementation.
        """
        from scipy import sparse as sp_sparse
        
        # Create 2x2 identity-like matrix for simple test
        # A = [[2, 0], [0, 3]]
        data = np.array([2, 3], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        A_cpu = sp_sparse.csr_matrix((data, indices, indptr), shape=(2, 2))
        
        # Transfer to GPU
        A_gpu = rb_matrix_gpu.from_cpu(A_cpu)
        
        # Multiply (naive implementation)
        with pytest.warns(UserWarning):  # Should warn about naive implementation
            C_gpu = A_gpu @ A_gpu
        
        # Transfer result back
        C_cpu = C_gpu.to_cpu()
        
        # Basic sanity check
        assert C_cpu.shape == (2, 2)
        assert C_cpu.nnz > 0


class TestMemoryTransfer:
    """Test memory transfer patterns between CPU and GPU."""
    
    def test_cupy_basic(self):
        """Verify CuPy is working correctly."""
        x_cpu = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x_gpu = cp.asarray(x_cpu)
        x_back = x_gpu.get()
        
        np.testing.assert_array_equal(x_cpu, x_back)
    
    def test_unified_memory_simulation(self):
        """
        Demonstrate unified memory concept (on DGX Spark this would be automatic).
        
        On non-UVM systems, this shows explicit transfer.
        On Grace Hopper with UVM, transfers are automatic.
        """
        # Allocate on CPU
        data_cpu = np.arange(100, dtype=np.int32)
        
        # Transfer to GPU
        data_gpu = cp.asarray(data_cpu)
        
        # Modify on GPU
        data_gpu *= 2
        
        # Transfer back
        data_result = data_gpu.get()
        
        # Validate
        np.testing.assert_array_equal(data_result, data_cpu * 2)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
