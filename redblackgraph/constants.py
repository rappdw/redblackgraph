"""
Red-Black Algebra Constants

The red-black semiring extends the natural numbers with two distinct multiplicative
identities based on parity:

- **BLACK_ONE (1)**: Identity for odd values in AVOS product
- **RED_ONE (-1)**: Identity for even values in AVOS product

AVOS Product Rules:
-------------------
The AVOS (Algebraic Vertex-Ordered Semiring) product (⊗) follows these rules:

  x ⊗ RED_ONE  = x if x is even, else 0 (undefined)
  x ⊗ BLACK_ONE = x if x is odd, else 0 (undefined)

In graph theory, these identities represent self-loops with different properties:
- RED_ONE typically appears on diagonal elements for vertices with even labels
- BLACK_ONE typically appears on diagonal elements for vertices with odd labels

NumPy Compatibility:
-------------------
For unsigned integer types, RED_ONE is represented as the maximum value (e.g., 255 
for uint8), which maintains the correct parity (always odd in binary representation).
"""

import numpy as np

# Multiplicative identities in the red-black semiring
BLACK_ONE = 1    # Identity for odd values
RED_ONE = -1     # Identity for even values (represented as maxint in unsigned types)


def red_one_for_dtype(dtype):
    """
    Get the RED_ONE value appropriate for a specific dtype.
    
    For signed integer types, RED_ONE is -1.
    For unsigned integer types, RED_ONE is the maximum value (e.g., 255 for uint8).
    This ensures correct behavior in both NumPy 1.x and 2.x.
    
    Parameters
    ----------
    dtype : numpy.dtype or type
        The target data type
        
    Returns
    -------
    int
        The RED_ONE value for the specified dtype
        
    Examples
    --------
    >>> red_one_for_dtype(np.int32)
    -1
    >>> red_one_for_dtype(np.uint8)
    255
    >>> red_one_for_dtype(np.uint32)
    4294967295
    """
    dt = np.dtype(dtype)
    if dt.kind == 'u':  # unsigned integer
        return np.iinfo(dt).max
    return RED_ONE


def black_one_for_dtype(dtype):
    """
    Get the BLACK_ONE value for a specific dtype.
    
    BLACK_ONE is always 1 regardless of dtype.
    This function is provided for symmetry with red_one_for_dtype.
    
    Parameters
    ----------
    dtype : numpy.dtype or type
        The target data type
        
    Returns
    -------
    int
        The BLACK_ONE value (always 1)
    """
    return BLACK_ONE


def is_red_one(value, dtype):
    """
    Check if a value represents RED_ONE for the given dtype.
    
    Parameters
    ----------
    value : int or array-like
        The value to check
    dtype : numpy.dtype or type
        The data type context
        
    Returns
    -------
    bool or array of bool
        True if value is RED_ONE for the dtype
    """
    dt = np.dtype(dtype)
    if dt.kind == 'u':
        return value == np.iinfo(dt).max
    return value == RED_ONE


def is_black_one(value):
    """
    Check if a value represents BLACK_ONE.
    
    Parameters
    ----------
    value : int or array-like
        The value to check
        
    Returns
    -------
    bool or array of bool
        True if value is BLACK_ONE (1)
    """
    return value == BLACK_ONE
