"""
Complete SpGEMM (Sparse General Matrix-Matrix Multiplication) for GPU.

This module provides the high-level interface for computing C = A @ B
using the two-phase SpGEMM algorithm with global memory hash tables:

1. Symbolic phase: Compute sparsity pattern using per-row hash tables
2. Numeric phase: Compute actual values using atomicMin for AVOS sum

Features:
- General A @ B and optimized A @ A (self-multiply)
- Optional upper triangular structure exploitation
- AVOS semiring operations
- Global memory hash tables (no arbitrary per-row limit)
- Deterministic output via global sort
- Memory-efficient CSR format
"""

from typing import Optional
from .csr_gpu import CSRMatrixGPU
from .spgemm_symbolic import compute_symbolic_pattern_with_tables
from .spgemm_numeric import compute_numeric_values

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def spgemm(
    A: CSRMatrixGPU,
    B: CSRMatrixGPU = None,
    *,
    upper_triangular: bool = None
) -> CSRMatrixGPU:
    """
    Compute C = A @ B using two-phase SpGEMM with AVOS semiring.

    Args:
        A: Left input CSR matrix
        B: Right input CSR matrix (if None, computes A @ A)
        upper_triangular: If True, apply mask j >= i on output.
            If None, automatically set to True when B is None and A.triangular.

    Returns:
        C: Result matrix

    Raises:
        ValueError: If matrices have incompatible shapes
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU operations")

    self_multiply = B is None or B is A
    if B is None:
        B = A

    # Shape validation
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes: A is {A.shape}, B is {B.shape}. "
            f"A.shape[1] must equal B.shape[0]."
        )

    # Infer upper_triangular if not specified
    if upper_triangular is None:
        upper_triangular = self_multiply and A.triangular

    n_rows = A.shape[0]
    n_cols = B.shape[1]

    # Phase 1: Symbolic - compute output pattern
    sym_kwargs = {}
    if not self_multiply:
        sym_kwargs['indptrB'] = B.indptr
        sym_kwargs['indicesB'] = B.indices

    indptrC, nnzC, hash_keys, table_offsets, table_sizes = compute_symbolic_pattern_with_tables(
        A.indptr, A.indices, n_rows, n_cols,
        upper_triangular=upper_triangular,
        **sym_kwargs
    )

    if nnzC == 0:
        return CSRMatrixGPU(
            cp.array([], dtype=cp.int32),
            cp.array([], dtype=cp.int32),
            indptrC,
            (n_rows, n_cols),
            triangular=upper_triangular,
            validate=False
        )

    # Phase 2: Numeric - compute values
    num_kwargs = {}
    if not self_multiply:
        num_kwargs['indptrB'] = B.indptr
        num_kwargs['indicesB'] = B.indices
        num_kwargs['dataB'] = B.data

    indicesC, dataC = compute_numeric_values(
        A.indptr, A.indices, A.data,
        indptrC, nnzC, n_rows,
        hash_keys, table_offsets, table_sizes,
        n_cols=n_cols,
        upper_triangular=upper_triangular,
        **num_kwargs
    )

    return CSRMatrixGPU(
        dataC,
        indicesC,
        indptrC,
        (n_rows, n_cols),
        triangular=upper_triangular,
        validate=False
    )


def spgemm_upper_triangular(
    A: CSRMatrixGPU,
    *,
    validate: bool = True
) -> CSRMatrixGPU:
    """
    Compute C = A @ A for upper triangular matrix A.

    This is a convenience wrapper around spgemm() for the common case
    of upper triangular self-multiplication.

    Args:
        A: Input CSR matrix (must be upper triangular)
        validate: If True, validate A is upper triangular

    Returns:
        C: Result matrix A @ A (also upper triangular)

    Raises:
        ValueError: If A is not square or not upper triangular
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got {A.shape}")

    if validate and not A.triangular:
        raise ValueError("Matrix must be upper triangular")

    return spgemm(A, upper_triangular=True)


def matmul_gpu(
    A: CSRMatrixGPU,
    B: Optional[CSRMatrixGPU] = None
) -> CSRMatrixGPU:
    """
    Matrix multiplication on GPU: C = A @ B.

    Args:
        A: First matrix
        B: Second matrix (if None, computes A @ A)

    Returns:
        C: Result matrix

    Raises:
        ValueError: If matrices have incompatible shapes
    """
    return spgemm(A, B)


class SpGEMMStats:
    """Statistics from SpGEMM operation."""

    def __init__(
        self,
        input_nnz: int,
        output_nnz: int,
        input_shape: tuple,
        symbolic_time: float = 0.0,
        numeric_time: float = 0.0
    ):
        self.input_nnz = input_nnz
        self.output_nnz = output_nnz
        self.input_shape = input_shape
        self.symbolic_time = symbolic_time
        self.numeric_time = numeric_time
        self.total_time = symbolic_time + numeric_time
        self.density_in = input_nnz / (input_shape[0] * input_shape[1]) if input_shape[0] * input_shape[1] > 0 else 0
        self.density_out = output_nnz / (input_shape[0] * input_shape[1]) if input_shape[0] * input_shape[1] > 0 else 0

    def __repr__(self) -> str:
        return (
            f"SpGEMMStats("
            f"input={self.input_nnz} nnz, "
            f"output={self.output_nnz} nnz, "
            f"time={self.total_time:.3f}s)"
        )


def spgemm_with_stats(A: CSRMatrixGPU, B: CSRMatrixGPU = None) -> tuple:
    """
    Compute C = A @ B and return statistics.

    Args:
        A: Left input matrix
        B: Right input matrix (if None, computes A @ A)

    Returns:
        C: Result matrix
        stats: SpGEMMStats object with timing information
    """
    import time

    self_multiply = B is None or B is A
    if B is None:
        B = A

    upper_triangular = self_multiply and A.triangular

    sym_kwargs = {}
    if not self_multiply:
        sym_kwargs['indptrB'] = B.indptr
        sym_kwargs['indicesB'] = B.indices

    # Symbolic phase
    start = time.perf_counter()
    indptrC, nnzC, hash_keys, table_offsets, table_sizes = compute_symbolic_pattern_with_tables(
        A.indptr, A.indices, A.shape[0], B.shape[1],
        upper_triangular=upper_triangular,
        **sym_kwargs
    )
    cp.cuda.Stream.null.synchronize()
    symbolic_time = time.perf_counter() - start

    # Numeric phase
    num_kwargs = {}
    if not self_multiply:
        num_kwargs['indptrB'] = B.indptr
        num_kwargs['indicesB'] = B.indices
        num_kwargs['dataB'] = B.data

    start = time.perf_counter()
    indicesC, dataC = compute_numeric_values(
        A.indptr, A.indices, A.data,
        indptrC, nnzC, A.shape[0],
        hash_keys, table_offsets, table_sizes,
        n_cols=B.shape[1],
        upper_triangular=upper_triangular,
        **num_kwargs
    )
    cp.cuda.Stream.null.synchronize()
    numeric_time = time.perf_counter() - start

    C = CSRMatrixGPU(
        dataC, indicesC, indptrC,
        (A.shape[0], B.shape[1]),
        triangular=upper_triangular,
        validate=False
    )

    stats = SpGEMMStats(
        input_nnz=A.nnz,
        output_nnz=nnzC,
        input_shape=A.shape,
        symbolic_time=symbolic_time,
        numeric_time=numeric_time
    )

    return C, stats
