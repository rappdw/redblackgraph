"""
Tests for GPU transitive closure via repeated squaring.

Validates against CPU transitive_closure_squaring() for bit-exact match.
"""

import numpy as np
import pytest
import scipy.sparse as sp

try:
    import cupy as cp
    from redblackgraph.gpu import CSRMatrixGPU, transitive_closure_gpu
    from redblackgraph.gpu.transitive_closure import sparse_avos_sum_gpu, sparse_equal_gpu
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")


def cpu_transitive_closure(A_dense):
    """CPU reference: transitive closure via repeated squaring using AVOS ops."""
    from redblackgraph.reference.rbg_math import avos_sum, avos_product

    n = A_dense.shape[0]
    R = A_dense.copy()

    for _ in range(64):
        # R_squared = R @ R (AVOS)
        R_sq = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                acc = 0
                for k in range(n):
                    p = avos_product(int(R[i, k]), int(R[k, j]))
                    acc = avos_sum(acc, p)
                R_sq[i, j] = acc

        # R_new = R ⊕ R_squared (element-wise AVOS sum)
        R_new = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                R_new[i, j] = avos_sum(int(R[i, j]), int(R_sq[i, j]))

        if np.array_equal(R, R_new):
            break
        R = R_new

    return R


class TestTransitiveClosureGPU:
    """Test GPU transitive closure against CPU reference."""

    def test_simple_chain(self):
        """Test: 0->1->2, closure should add 0->2."""
        A = np.array([
            [1, 2, 0],
            [0, -1, 3],
            [0, 0, 1],
        ], dtype=np.int32)

        expected = cpu_transitive_closure(A)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, diameter = A_gpu.transitive_closure()
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, expected)

    def test_identity_already_closed(self):
        """Identity matrix is already its own closure."""
        A = np.eye(4, dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, diameter = transitive_closure_gpu(A_gpu)
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, A)

    def test_empty_matrix(self):
        """Empty matrix closure is empty."""
        A = sp.csr_matrix((3, 3), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=True)
        R_gpu, diameter = transitive_closure_gpu(A_gpu)
        assert R_gpu.nnz == 0

    def test_two_hops(self):
        """4-vertex chain: 0->1->2->3, should find 0->2, 0->3, 1->3."""
        A = np.array([
            [1, 2, 0, 0],
            [0, -1, 3, 0],
            [0, 0, 1, 2],
            [0, 0, 0, -1],
        ], dtype=np.int32)

        expected = cpu_transitive_closure(A)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, _ = transitive_closure_gpu(A_gpu)
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, expected)

    def test_diamond_dag(self):
        """Diamond: 0->{1,2}->3."""
        A = np.array([
            [1, 2, 3, 0],
            [0, -1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, -1],
        ], dtype=np.int32)

        expected = cpu_transitive_closure(A)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, _ = transitive_closure_gpu(A_gpu)
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, expected)

    def test_disconnected_components(self):
        """Two disconnected pairs: {0->1}, {2->3}."""
        A = np.array([
            [1, 2, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 3],
            [0, 0, 0, -1],
        ], dtype=np.int32)

        expected = cpu_transitive_closure(A)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, _ = transitive_closure_gpu(A_gpu)
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, expected)

    def test_matches_cpu_squaring(self):
        """Compare against sparse backend's transitive_closure_squaring."""
        from redblackgraph.sparse.rbm import rb_matrix
        from redblackgraph.sparse.csgraph.transitive_closure import transitive_closure_squaring

        A = np.array([
            [1, 2, 0, 0, 0],
            [0, -1, 3, 0, 0],
            [0, 0, 1, 2, 3],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.int32)

        # CPU reference
        A_rb = rb_matrix(sp.csr_matrix(A))
        tc_cpu = transitive_closure_squaring(A_rb)
        expected = tc_cpu.W.toarray()

        # GPU
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        R_gpu, _ = transitive_closure_gpu(A_gpu)
        result = R_gpu.to_cpu().toarray()

        assert np.array_equal(result, expected)


class TestSparseAvosSum:
    """Test sparse AVOS sum (element-wise merge)."""

    def test_disjoint_patterns(self):
        """Merging matrices with no overlapping entries."""
        A = sp.csr_matrix(np.array([[1, 0], [0, 0]], dtype=np.int32))
        B = sp.csr_matrix(np.array([[0, 0], [0, 2]], dtype=np.int32))

        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(B, triangular=False)
        C_gpu = sparse_avos_sum_gpu(A_gpu, B_gpu)
        result = C_gpu.to_cpu().toarray()

        expected = np.array([[1, 0], [0, 2]], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_overlapping_min(self):
        """Overlapping entries should take minimum absolute value."""
        A = sp.csr_matrix(np.array([[4, 0], [0, 6]], dtype=np.int32))
        B = sp.csr_matrix(np.array([[2, 0], [0, 3]], dtype=np.int32))

        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(B, triangular=False)
        C_gpu = sparse_avos_sum_gpu(A_gpu, B_gpu)
        result = C_gpu.to_cpu().toarray()

        expected = np.array([[2, 0], [0, 3]], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_empty_a(self):
        """A empty, result is B."""
        A = sp.csr_matrix((3, 3), dtype=np.int32)
        B = sp.csr_matrix(np.eye(3, dtype=np.int32))

        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(B, triangular=False)
        C_gpu = sparse_avos_sum_gpu(A_gpu, B_gpu)
        result = C_gpu.to_cpu().toarray()

        assert np.array_equal(result, B.toarray())


class TestSparseEqual:
    """Test sparse equality check."""

    def test_equal(self):
        A = sp.csr_matrix(np.array([[1, 2], [0, 3]], dtype=np.int32))
        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        assert sparse_equal_gpu(A_gpu, B_gpu)

    def test_not_equal_data(self):
        A = sp.csr_matrix(np.array([[1, 2], [0, 3]], dtype=np.int32))
        B = sp.csr_matrix(np.array([[1, 2], [0, 4]], dtype=np.int32))
        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(B, triangular=False)
        assert not sparse_equal_gpu(A_gpu, B_gpu)

    def test_not_equal_nnz(self):
        A = sp.csr_matrix(np.array([[1, 0], [0, 3]], dtype=np.int32))
        B = sp.csr_matrix(np.array([[1, 2], [0, 3]], dtype=np.int32))
        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(B, triangular=False)
        assert not sparse_equal_gpu(A_gpu, B_gpu)

    def test_both_empty(self):
        A = sp.csr_matrix((3, 3), dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        B_gpu = CSRMatrixGPU.from_cpu(A, triangular=False)
        assert sparse_equal_gpu(A_gpu, B_gpu)


class TestCSRMatrixGPUOperators:
    """Test the new CSRMatrixGPU operators."""

    def test_matmul_operator(self):
        """Test A @ B via __matmul__."""
        A = np.array([
            [1, 2, 0],
            [0, -1, 3],
            [0, 0, 1],
        ], dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        C_gpu = A_gpu @ A_gpu
        assert C_gpu.nnz > 0

    def test_copy(self):
        A = np.eye(3, dtype=np.int32)
        A_gpu = CSRMatrixGPU.from_cpu(sp.csr_matrix(A), triangular=True)
        B_gpu = A_gpu.copy()
        assert sparse_equal_gpu(A_gpu, B_gpu)
        # Modifying copy shouldn't affect original
        B_gpu.data[0] = 99
        assert not sparse_equal_gpu(A_gpu, B_gpu)

    def test_eliminate_zeros(self):
        # Create matrix with explicit zeros
        data = cp.array([1, 0, 3], dtype=cp.int32)
        indices = cp.array([0, 1, 2], dtype=cp.int32)
        indptr = cp.array([0, 1, 2, 3], dtype=cp.int32)
        A = CSRMatrixGPU(data, indices, indptr, (3, 3), triangular=True, validate=False)
        assert A.nnz == 3
        A.eliminate_zeros()
        assert A.nnz == 2
