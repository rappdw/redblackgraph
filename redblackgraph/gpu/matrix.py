"""
GPU-accelerated rb_matrix using CuPy sparse matrices.

Provides a high-level matrix wrapper that routes ``@`` to the production
SpGEMM pipeline for upper-triangular matrices and falls back to a naive
dense path for small non-triangular matrices.
"""

import warnings

import numpy as np

from ._cuda_utils import CUPY_AVAILABLE, check_cupy

if CUPY_AVAILABLE:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
else:
    cp = None
    cp_sparse = None


class rb_matrix_gpu:
    """
    GPU-accelerated Red-Black matrix using CuPy sparse CSR format.

    Features:
    - Wraps cupyx.scipy.sparse.csr_matrix
    - Supports triangular=True flag for upper triangular matrices
    - Routes ``@`` to production SpGEMM for triangular self-multiplication
    - Falls back to naive dense path for small non-triangular matrices
    - Provides to_cpu() / from_cpu() / to_csr_gpu() conversions
    """

    def __init__(self, data=None, *, triangular=False, shape=None):
        """
        Initialize GPU rb_matrix.

        Args:
            data: Can be:
                - cupyx.scipy.sparse matrix
                - CPU scipy.sparse matrix (will be transferred)
                - (data, indices, indptr) tuple for CSR format
                - None (creates empty matrix, requires shape)
            triangular: If True, matrix is upper triangular
            shape: Shape tuple (required if data is None)
        """
        check_cupy()

        self.triangular = triangular

        if data is None:
            if shape is None:
                raise ValueError("shape required when data is None")
            self.data = cp_sparse.csr_matrix(shape, dtype=cp.float32)
        elif isinstance(data, cp_sparse.csr_matrix):
            self.data = data
        elif hasattr(data, 'tocsr'):
            if hasattr(data, 'get'):
                self.data = data.tocsr()
            else:
                cpu_csr = data.tocsr()
                self.data = cp_sparse.csr_matrix(
                    (cp.array(cpu_csr.data, dtype=cp.float32),
                     cp.array(cpu_csr.indices, dtype=cp.int32),
                     cp.array(cpu_csr.indptr, dtype=cp.int32)),
                    shape=cpu_csr.shape
                )
        elif isinstance(data, tuple) and len(data) == 3:
            data_arr, indices_arr, indptr_arr = data
            self.data = cp_sparse.csr_matrix(
                (cp.asarray(data_arr, dtype=cp.float32),
                 cp.asarray(indices_arr, dtype=cp.int32),
                 cp.asarray(indptr_arr, dtype=cp.int32)),
                shape=shape
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @property
    def shape(self):
        return self.data.shape

    @property
    def nnz(self):
        return self.data.nnz

    def to_cpu(self):
        """Transfer matrix to CPU as scipy.sparse.csr_matrix."""
        from scipy import sparse as sp_sparse
        return sp_sparse.csr_matrix(
            (self.data.data.get(),
             self.data.indices.get(),
             self.data.indptr.get()),
            shape=self.shape
        )

    def to_csr_gpu(self):
        """Convert to production CSRMatrixGPU (int32 data).

        Returns:
            CSRMatrixGPU suitable for use with spgemm_upper_triangular().
        """
        from .csr_gpu import CSRMatrixGPU

        cpu_csr = self.to_cpu()
        cpu_csr.data = cpu_csr.data.astype(np.int32)
        return CSRMatrixGPU.from_cpu(cpu_csr, triangular=self.triangular)

    @classmethod
    def from_cpu(cls, cpu_matrix, triangular=False):
        """Create GPU matrix from CPU sparse matrix."""
        return cls(cpu_matrix, triangular=triangular)

    @classmethod
    def from_csr_gpu(cls, csr_gpu):
        """Create rb_matrix_gpu from a CSRMatrixGPU.

        Args:
            csr_gpu: CSRMatrixGPU instance (int32 data)

        Returns:
            rb_matrix_gpu wrapping a CuPy sparse matrix
        """
        cpu_csr = csr_gpu.to_cpu()
        return cls(cpu_csr, triangular=csr_gpu.triangular)

    def __matmul__(self, other):
        """Matrix multiplication using AVOS operations.

        For upper-triangular self-multiplication (A @ A), routes to the
        production SpGEMM pipeline. Falls back to a naive dense path for
        small non-triangular matrices.
        """
        if not isinstance(other, rb_matrix_gpu):
            raise TypeError("Can only multiply rb_matrix_gpu with rb_matrix_gpu")

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")

        # Fast path: triangular self-multiply via production SpGEMM
        if self.triangular and other.triangular:
            return self._spgemm_matmul(other)

        # Slow path: naive dense fallback
        return self._naive_matmul(other)

    def _spgemm_matmul(self, other):
        """Route to production SpGEMM pipeline."""
        from .spgemm import spgemm_upper_triangular, matmul_gpu

        a_csr = self.to_csr_gpu()

        if other is self:
            c_csr = spgemm_upper_triangular(a_csr)
        else:
            b_csr = other.to_csr_gpu()
            c_csr = matmul_gpu(a_csr, b_csr)

        return rb_matrix_gpu.from_csr_gpu(c_csr)

    def _naive_matmul(self, other):
        """Naive dense matrix multiplication (deprecated fallback)."""
        from .core import avos_sum_gpu as _naive_sum, avos_product_gpu as _naive_prod

        warnings.warn(
            "Using naive dense matrix multiplication for non-triangular matrices. "
            "This is slow and will be removed in a future release. "
            "Convert to CSRMatrixGPU and use spgemm_upper_triangular() instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        if self.nnz > 10000 or other.nnz > 10000:
            raise NotImplementedError(
                "Naive implementation limited to small matrices (nnz < 10k). "
                "Use optimized SpGEMM kernels for production."
            )

        A_dense = self.data.toarray()
        B_dense = other.data.toarray()
        C_dense = cp.zeros((self.shape[0], other.shape[1]), dtype=cp.float32)

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        for i in range(m):
            for j in range(p):
                if self.triangular and j < i:
                    continue
                acc = cp.float32(0)
                for k in range(n):
                    if A_dense[i, k] != 0 and B_dense[k, j] != 0:
                        prod = _naive_prod(A_dense[i, k], B_dense[k, j])
                        acc = _naive_sum(acc, prod)
                C_dense[i, j] = acc

        C_sparse = cp_sparse.csr_matrix(C_dense)
        return rb_matrix_gpu(C_sparse, triangular=self.triangular)

    def __repr__(self):
        return (f"rb_matrix_gpu(shape={self.shape}, nnz={self.nnz}, "
                f"triangular={self.triangular}, device='gpu')")
