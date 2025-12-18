"""
Complete SpGEMM (Sparse General Matrix-Matrix Multiplication) for GPU.

This module provides the high-level interface for computing C = A @ A
using the two-phase SpGEMM algorithm with global memory hash tables:

1. Symbolic phase: Compute sparsity pattern using per-row hash tables
2. Numeric phase: Compute actual values using atomicMin for AVOS sum

Features:
- Upper triangular structure exploitation
- AVOS semiring operations
- Global memory hash tables (no arbitrary per-row limit)
- Deterministic output via global sort
- Memory-efficient CSR format
"""

from typing import Optional
from .csr_gpu import CSRMatrixGPU
from .spgemm_symbolic import compute_symbolic_pattern
from .spgemm_numeric import compute_numeric_values

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def spgemm_upper_triangular(
    A: CSRMatrixGPU,
    *,
    validate: bool = True
) -> CSRMatrixGPU:
    """
    Compute C = A @ A for upper triangular matrix A.

    This is the main entry point for sparse matrix multiplication on GPU.
    Uses two-phase SpGEMM algorithm:

    1. Symbolic: Determine output pattern (which entries are non-zero)
    2. Numeric: Compute actual values using AVOS operations

    Args:
        A: Input CSR matrix (must be upper triangular)
        validate: If True, validate A is upper triangular

    Returns:
        C: Result matrix A @ A (also upper triangular)

    Raises:
        ValueError: If A is not square or not upper triangular

    Example:
        >>> A_cpu = scipy.sparse.csr_matrix(...)
        >>> A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        >>> C_gpu = spgemm_upper_triangular(A_gpu)
        >>> C_cpu = C_gpu.to_cpu()
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU operations")

    # Validate input
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got {A.shape}")

    if validate and not A.triangular:
        raise ValueError("Matrix must be upper triangular")

    n_rows, n_cols = A.shape

    # Phase 1: Symbolic - compute output pattern using global memory hash tables
    indptrC, nnzC, hash_keys, table_offsets, table_sizes = compute_symbolic_pattern(
        A.indptr, A.indices, n_rows, n_cols
    )

    if nnzC == 0:
        # Empty result
        return CSRMatrixGPU(
            cp.array([], dtype=cp.int32),
            cp.array([], dtype=cp.int32),
            indptrC,
            A.shape,
            triangular=True,
            validate=False
        )

    # Phase 2: Numeric - compute values using hash tables from symbolic phase
    indicesC, dataC = compute_numeric_values(
        A.indptr, A.indices, A.data,
        indptrC, nnzC, n_rows,
        hash_keys, table_offsets, table_sizes
    )

    # Create result matrix
    C = CSRMatrixGPU(
        dataC,
        indicesC,
        indptrC,
        A.shape,
        triangular=True,
        validate=validate
    )

    return C


def matmul_gpu(
    A: CSRMatrixGPU,
    B: Optional[CSRMatrixGPU] = None
) -> CSRMatrixGPU:
    """
    Matrix multiplication on GPU: C = A @ B.

    Currently optimized for the special case B = A (self-multiplication)
    with upper triangular structure.

    Args:
        A: First matrix
        B: Second matrix (if None, computes A @ A)

    Returns:
        C: Result matrix

    Raises:
        ValueError: If matrices have incompatible shapes
        NotImplementedError: If B is provided and differs from A
    """
    if B is not None and B is not A:
        raise NotImplementedError(
            "General A @ B not yet implemented. "
            "Currently only supports A @ A (self-multiplication)."
        )

    if not A.triangular:
        raise NotImplementedError(
            "General (non-triangular) matrices not yet implemented. "
            "Currently only supports upper triangular matrices."
        )

    return spgemm_upper_triangular(A)


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
        """
        Initialize statistics.

        Args:
            input_nnz: Non-zeros in input matrix
            output_nnz: Non-zeros in output matrix
            input_shape: Shape of input matrix
            symbolic_time: Time for symbolic phase (seconds)
            numeric_time: Time for numeric phase (seconds)
        """
        self.input_nnz = input_nnz
        self.output_nnz = output_nnz
        self.input_shape = input_shape
        self.symbolic_time = symbolic_time
        self.numeric_time = numeric_time
        self.total_time = symbolic_time + numeric_time
        self.density_in = input_nnz / (input_shape[0] * input_shape[1])
        self.density_out = output_nnz / (input_shape[0] * input_shape[1])

    def __repr__(self) -> str:
        return (
            f"SpGEMMStats("
            f"input={self.input_nnz} nnz, "
            f"output={self.output_nnz} nnz, "
            f"time={self.total_time:.3f}s)"
        )


def spgemm_with_stats(A: CSRMatrixGPU) -> tuple:
    """
    Compute C = A @ A and return statistics.

    Useful for performance analysis and optimization.

    Args:
        A: Input matrix

    Returns:
        C: Result matrix
        stats: SpGEMMStats object with timing information
    """
    import time

    # Symbolic phase - compute output pattern using global memory hash tables
    start = time.perf_counter()
    indptrC, nnzC, hash_keys, table_offsets, table_sizes = compute_symbolic_pattern(
        A.indptr, A.indices, A.shape[0], A.shape[1]
    )
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    symbolic_time = time.perf_counter() - start

    # Numeric phase - compute values using hash tables from symbolic phase
    start = time.perf_counter()
    indicesC, dataC = compute_numeric_values(
        A.indptr, A.indices, A.data,
        indptrC, nnzC, A.shape[0],
        hash_keys, table_offsets, table_sizes
    )
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    numeric_time = time.perf_counter() - start

    # Create result
    C = CSRMatrixGPU(
        dataC, indicesC, indptrC,
        A.shape, triangular=True, validate=False
    )

    # Create stats
    stats = SpGEMMStats(
        input_nnz=A.nnz,
        output_nnz=nnzC,
        input_shape=A.shape,
        symbolic_time=symbolic_time,
        numeric_time=numeric_time
    )

    return C, stats
