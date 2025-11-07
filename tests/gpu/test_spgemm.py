"""
Tests for complete SpGEMM (sparse matrix-matrix multiplication) on GPU.

Tests the full two-phase algorithm with AVOS operations.
"""

import pytest
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    from redblackgraph.gpu.csr_gpu import CSRMatrixGPU
    from redblackgraph.gpu.spgemm import (
        spgemm_upper_triangular,
        matmul_gpu,
        spgemm_with_stats
    )
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")

# Import CPU reference for validation
from redblackgraph.reference.rbg_math import avos_sum, avos_product


def compute_spgemm_cpu(A_csr):
    """
    CPU reference: compute C = A @ A using AVOS operations.
    
    Returns scipy CSR matrix with int32 data.
    """
    n = A_csr.shape[0]
    
    # Use dense for simplicity in reference
    A_dense = A_csr.toarray()
    C_dense = np.zeros((n, n), dtype=np.int32)
    
    for i in range(n):
        for j in range(i, n):  # Upper triangular only
            acc = 0
            for k in range(n):
                if A_dense[i, k] != 0 and A_dense[k, j] != 0:
                    prod = avos_product(int(A_dense[i, k]), int(A_dense[k, j]))
                    acc = avos_sum(acc, prod)
            C_dense[i, j] = acc
    
    # Convert to sparse
    return sp.csr_matrix(C_dense, dtype=np.int32)


class TestIdentityMatrix:
    """Test with identity matrix (simplest case)."""
    
    def test_identity_3x3(self):
        """Test I @ I = I for 3x3 identity."""
        # Create identity matrix
        I_cpu = sp.eye(3, dtype=np.int32, format='csr')
        I_gpu = CSRMatrixGPU.from_cpu(I_cpu, triangular=True)
        
        # Compute I @ I
        C_gpu = spgemm_upper_triangular(I_gpu)
        C_cpu = C_gpu.to_cpu()
        
        # Should equal identity
        assert C_cpu.shape == (3, 3)
        assert C_cpu.nnz == 3
        assert np.array_equal(C_cpu.toarray(), I_cpu.toarray())
    
    def test_identity_10x10(self):
        """Test larger identity matrix."""
        I_cpu = sp.eye(10, dtype=np.int32, format='csr')
        I_gpu = CSRMatrixGPU.from_cpu(I_cpu, triangular=True)
        
        C_gpu = spgemm_upper_triangular(I_gpu)
        C_cpu = C_gpu.to_cpu()
        
        assert C_cpu.shape == (10, 10)
        assert C_cpu.nnz == 10
        assert np.array_equal(C_cpu.toarray(), I_cpu.toarray())


class TestSimpleTriangular:
    """Test with simple upper triangular matrices."""
    
    def test_2x2_simple(self):
        """Test 2x2 upper triangular."""
        # [2 3]
        # [0 4]
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 1])
        data = np.array([2, 3, 4], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(2, 2))
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        # Compute reference
        C_ref = compute_spgemm_cpu(A_cpu)
        
        # Compare
        assert C_cpu.shape == C_ref.shape
        assert np.array_equal(C_cpu.toarray(), C_ref.toarray())
    
    def test_3x3_triangular(self):
        """Test 3x3 upper triangular."""
        # [2  3  0]
        # [0  4  5]
        # [0  0  6]
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        # Compute reference
        C_ref = compute_spgemm_cpu(A_cpu)
        
        # Compare shapes and nnz
        assert C_cpu.shape == C_ref.shape
        
        # Compare values (element by element for debugging)
        C_cpu_dense = C_cpu.toarray()
        C_ref_dense = C_ref.toarray()
        
        for i in range(3):
            for j in range(3):
                assert C_cpu_dense[i, j] == C_ref_dense[i, j], \
                    f"Mismatch at ({i},{j}): GPU={C_cpu_dense[i,j]}, CPU={C_ref_dense[i,j]}"


class TestDiagonalMatrix:
    """Test with diagonal matrices."""
    
    def test_diagonal_5x5(self):
        """Test diagonal matrix multiplication."""
        # Diagonal with values [2, 3, 4, 5, 6]
        data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        A_cpu = sp.diags(data, 0, format='csr', dtype=np.int32)
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        # Diagonal @ Diagonal = squared diagonal
        C_ref = compute_spgemm_cpu(A_cpu)
        
        assert np.array_equal(C_cpu.toarray(), C_ref.toarray())


class TestEmptyAndSparse:
    """Test edge cases with empty rows and very sparse matrices."""
    
    def test_empty_matrix(self):
        """Test with completely empty matrix."""
        A_cpu = sp.csr_matrix((5, 5), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        
        C_gpu = spgemm_upper_triangular(A_gpu)
        
        assert C_gpu.nnz == 0
        assert C_gpu.shape == (5, 5)
    
    def test_single_nonzero(self):
        """Test with single non-zero element."""
        # Matrix with single element at (0, 0)
        A_cpu = sp.csr_matrix(([42], ([0], [0])), shape=(5, 5), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        C_ref = compute_spgemm_cpu(A_cpu)
        assert np.array_equal(C_cpu.toarray(), C_ref.toarray())
    
    def test_sparse_with_empty_rows(self):
        """Test sparse matrix with some empty rows."""
        # [2  0  0  0]
        # [0  0  0  0]  <- empty
        # [0  0  3  0]
        # [0  0  0  4]
        row = np.array([0, 2, 3])
        col = np.array([0, 2, 3])
        data = np.array([2, 3, 4], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(4, 4))
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        C_ref = compute_spgemm_cpu(A_cpu)
        assert np.array_equal(C_cpu.toarray(), C_ref.toarray())


class TestAVOSSemantics:
    """Test that AVOS operations are correctly applied."""
    
    def test_avos_identities(self):
        """Test with RED_ONE and BLACK_ONE identities."""
        # Matrix with identity values
        # [-1  2]
        # [ 0  1]
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 1])
        data = np.array([-1, 2, 1], dtype=np.int32)  # RED_ONE, value, BLACK_ONE
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(2, 2))
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        C_cpu = C_gpu.to_cpu()
        
        C_ref = compute_spgemm_cpu(A_cpu)
        
        # Should handle identities correctly
        assert C_cpu.shape == C_ref.shape
        assert np.array_equal(C_cpu.toarray(), C_ref.toarray())


class TestMatmulAPI:
    """Test high-level matmul_gpu API."""
    
    def test_matmul_self(self):
        """Test matmul_gpu(A) computes A @ A."""
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 1, 2])
        data = np.array([2, 3, 4, 5], dtype=np.int32)
        A_cpu = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        
        # Using matmul_gpu
        C1 = matmul_gpu(A_gpu)
        
        # Using spgemm_upper_triangular directly
        C2 = spgemm_upper_triangular(A_gpu)
        
        # Should be identical
        assert np.array_equal(C1.to_cpu().toarray(), C2.to_cpu().toarray())
    
    def test_matmul_errors(self):
        """Test that unsupported operations raise errors."""
        A_cpu = sp.eye(3, dtype=np.int32, format='csr')
        B_cpu = sp.eye(3, dtype=np.int32, format='csr') * 2
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        B_gpu = CSRMatrixGPU.from_cpu(B_cpu, triangular=True)
        
        # A @ B (different matrices) not yet implemented
        with pytest.raises(NotImplementedError):
            matmul_gpu(A_gpu, B_gpu)


class TestSpGEMMStats:
    """Test statistics collection."""
    
    def test_stats_collection(self):
        """Test that statistics are collected correctly."""
        A_cpu = sp.eye(10, dtype=np.int32, format='csr')
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        
        C, stats = spgemm_with_stats(A_gpu)
        
        assert stats.input_nnz == 10
        assert stats.output_nnz == 10
        assert stats.input_shape == (10, 10)
        assert stats.symbolic_time >= 0
        assert stats.numeric_time >= 0
        assert stats.total_time >= 0
        assert stats.density_in == 0.1
        assert stats.density_out == 0.1


class TestValidation:
    """Test input validation."""
    
    def test_non_square_error(self):
        """Test that non-square matrices are rejected."""
        A_cpu = sp.csr_matrix((3, 4), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=False, validate=False)
        
        with pytest.raises(ValueError, match="square"):
            spgemm_upper_triangular(A_gpu)
    
    def test_non_triangular_error(self):
        """Test that non-triangular matrices are rejected."""
        # Lower triangular element
        A_cpu = sp.csr_matrix(([1], ([1], [0])), shape=(3, 3), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=False, validate=False)
        
        with pytest.raises(ValueError, match="triangular"):
            spgemm_upper_triangular(A_gpu, validate=True)


@pytest.mark.slow
class TestLargerMatrices:
    """Test with larger matrices (marked slow)."""
    
    def test_50x50_sparse(self):
        """Test 50x50 sparse matrix."""
        n = 50
        density = 0.1
        
        # Generate random upper triangular
        A_cpu = sp.random(n, n, density=density, format='csr', dtype=np.int32)
        
        # Make upper triangular
        A_cpu = sp.triu(A_cpu, format='csr')
        A_cpu.data = A_cpu.data.astype(np.int32)
        
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        C_gpu = spgemm_upper_triangular(A_gpu)
        
        # Just verify it completes and has reasonable output
        assert C_gpu.shape == (n, n)
        assert C_gpu.nnz > 0
        assert C_gpu.triangular
