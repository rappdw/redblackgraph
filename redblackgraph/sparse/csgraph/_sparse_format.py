"""
Sparse Format Conversion Utilities for Red-Black Graphs

Provides wrappers around scipy.sparse utilities with RBG-specific handling
for rb_matrix and rb_array types.
"""

import numpy as np
from scipy.sparse import (
    csr_matrix, csc_matrix, coo_matrix,
    isspmatrix, isspmatrix_csr, isspmatrix_csc, isspmatrix_coo
)
from typing import Union, Optional, Tuple
import warnings


# Type alias for sparse matrix types we support
SparseMatrix = Union[csr_matrix, csc_matrix, coo_matrix]


def is_sparse(A) -> bool:
    """
    Check if A is a scipy sparse matrix.
    
    Parameters
    ----------
    A : array-like
        Input matrix
        
    Returns
    -------
    bool
        True if A is a scipy sparse matrix
    """
    return isspmatrix(A)


def is_dense(A) -> bool:
    """
    Check if A is a dense numpy array or similar.
    
    Parameters
    ----------
    A : array-like
        Input matrix
        
    Returns
    -------
    bool
        True if A is a dense array (numpy ndarray or similar)
    """
    return isinstance(A, np.ndarray) or hasattr(A, '__array__') and not isspmatrix(A)


def get_density(A) -> float:
    """
    Compute the density of a matrix (fraction of non-zero entries).
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
        
    Returns
    -------
    float
        Density value between 0 and 1
    """
    if isspmatrix(A):
        n_elements = A.shape[0] * A.shape[1]
        if n_elements == 0:
            return 0.0
        return A.nnz / n_elements
    else:
        arr = np.asarray(A)
        n_elements = arr.size
        if n_elements == 0:
            return 0.0
        return np.count_nonzero(arr) / n_elements


def get_nnz(A) -> int:
    """
    Get the number of non-zero entries in a matrix.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
        
    Returns
    -------
    int
        Number of non-zero entries
    """
    if isspmatrix(A):
        return A.nnz
    else:
        return int(np.count_nonzero(np.asarray(A)))


def ensure_csr(A, copy: bool = False) -> csr_matrix:
    """
    Convert matrix to CSR format if needed.
    
    Handles:
    - scipy sparse matrices (converts format if needed)
    - numpy arrays (converts to sparse)
    - rb_array/rb_matrix types (extracts underlying array)
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
    copy : bool, default False
        If True, always return a copy
        
    Returns
    -------
    csr_matrix
        Matrix in CSR format
    """
    # Handle rb_array/rb_matrix by extracting underlying array
    if hasattr(A, 'view') and not isspmatrix(A):
        # rb_array wraps numpy array, get the view
        A = np.asarray(A)
    
    if isspmatrix_csr(A):
        return A.copy() if copy else A
    elif isspmatrix(A):
        return A.tocsr(copy=copy)
    else:
        # Dense array - convert to CSR
        return csr_matrix(np.asarray(A))


def ensure_csc(A, copy: bool = False) -> csc_matrix:
    """
    Convert matrix to CSC format if needed.
    
    CSC format is efficient for column-wise operations (e.g., finding
    all incoming edges to a vertex).
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
    copy : bool, default False
        If True, always return a copy
        
    Returns
    -------
    csc_matrix
        Matrix in CSC format
    """
    # Handle rb_array/rb_matrix by extracting underlying array
    if hasattr(A, 'view') and not isspmatrix(A):
        A = np.asarray(A)
    
    if isspmatrix_csc(A):
        return A.copy() if copy else A
    elif isspmatrix(A):
        return A.tocsc(copy=copy)
    else:
        # Dense array - convert to CSC
        return csc_matrix(np.asarray(A))


def ensure_coo(A, copy: bool = False) -> coo_matrix:
    """
    Convert matrix to COO format if needed.
    
    COO format is efficient for constructing sparse matrices incrementally.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
    copy : bool, default False
        If True, always return a copy
        
    Returns
    -------
    coo_matrix
        Matrix in COO format
    """
    if hasattr(A, 'view') and not isspmatrix(A):
        A = np.asarray(A)
    
    if isspmatrix_coo(A):
        return A.copy() if copy else A
    elif isspmatrix(A):
        return A.tocoo(copy=copy)
    else:
        return coo_matrix(np.asarray(A))


def to_dense(A) -> np.ndarray:
    """
    Convert sparse matrix to dense numpy array.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input matrix
        
    Returns
    -------
    np.ndarray
        Dense numpy array
    """
    if isspmatrix(A):
        return A.toarray()
    else:
        return np.asarray(A)


def to_dense_if_needed(
    A,
    density_threshold: float = 0.5,
    size_threshold: int = 1000,
    warn: bool = True
) -> Union[np.ndarray, SparseMatrix]:
    """
    Conditionally convert to dense if the matrix is small or dense enough.
    
    This is useful for algorithms that may be faster on dense matrices
    for small or dense inputs.
    
    Parameters
    ----------
    A : sparse matrix or array-like
        Input matrix
    density_threshold : float, default 0.5
        Convert to dense if density exceeds this threshold
    size_threshold : int, default 1000
        Convert to dense if matrix dimension is below this threshold
    warn : bool, default True
        Emit warning when densifying
        
    Returns
    -------
    array-like
        Dense array if conditions met, otherwise original sparse matrix
    """
    if not isspmatrix(A):
        return np.asarray(A)
    
    n = max(A.shape)
    density = get_density(A)
    
    should_densify = (n < size_threshold) or (density > density_threshold)
    
    if should_densify:
        if warn and n >= 100:  # Only warn for non-trivial matrices
            warnings.warn(
                f"Densifying sparse matrix (n={n}, density={density:.3f}). "
                f"This may use O(n²) memory.",
                stacklevel=2
            )
        return A.toarray()
    
    return A


def get_format(A) -> str:
    """
    Get the format string for a sparse matrix.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
        
    Returns
    -------
    str
        Format string: 'csr', 'csc', 'coo', 'dense', or 'unknown'
    """
    if isspmatrix_csr(A):
        return 'csr'
    elif isspmatrix_csc(A):
        return 'csc'
    elif isspmatrix_coo(A):
        return 'coo'
    elif isspmatrix(A):
        return A.format if hasattr(A, 'format') else 'unknown'
    elif isinstance(A, np.ndarray) or hasattr(A, '__array__'):
        return 'dense'
    else:
        return 'unknown'


def csr_to_csc_transpose(A_csr: csr_matrix) -> csc_matrix:
    """
    Get CSC format of transpose efficiently.
    
    For a CSR matrix A, A.T in CSC format shares the same underlying
    data structure, so this is O(1).
    
    Parameters
    ----------
    A_csr : csr_matrix
        Input matrix in CSR format
        
    Returns
    -------
    csc_matrix
        Transpose of A in CSC format
    """
    # CSR(A).T == CSC(A.T), but scipy handles this efficiently
    return A_csr.T.tocsc()


def csr_csc_pair(A) -> Tuple[csr_matrix, csc_matrix]:
    """
    Get both CSR and CSC representations of a matrix.
    
    This is useful when both row and column iteration are needed,
    such as in component finding (outgoing and incoming edges).
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix
        
    Returns
    -------
    tuple of (csr_matrix, csc_matrix)
        CSR and CSC representations
    """
    A_csr = ensure_csr(A)
    A_csc = A_csr.tocsc()
    return A_csr, A_csc


def empty_csr(n: int, dtype=np.int32) -> csr_matrix:
    """
    Create an empty n×n CSR matrix.
    
    Parameters
    ----------
    n : int
        Matrix dimension
    dtype : dtype, default np.int32
        Data type
        
    Returns
    -------
    csr_matrix
        Empty sparse matrix
    """
    return csr_matrix((n, n), dtype=dtype)


def identity_csr(n: int, dtype=np.int32, diagonal_value: int = 1) -> csr_matrix:
    """
    Create an n×n identity matrix in CSR format.
    
    Parameters
    ----------
    n : int
        Matrix dimension
    dtype : dtype, default np.int32
        Data type
    diagonal_value : int, default 1
        Value to place on diagonal (use -1 for RED_ONE identity, 1 for BLACK_ONE)
        
    Returns
    -------
    csr_matrix
        Identity matrix in CSR format
    """
    from scipy.sparse import eye
    if diagonal_value == 1:
        return eye(n, dtype=dtype, format='csr')
    else:
        return csr_matrix((np.full(n, diagonal_value, dtype=dtype), 
                          (np.arange(n), np.arange(n))), shape=(n, n))
