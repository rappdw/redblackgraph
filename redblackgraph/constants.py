"""
Red-Black Algebra Constants

The red-black semiring extends the natural numbers with two distinct multiplicative
identities based on parity. This algebraic structure emerges naturally from modeling
binary trees and genealogical relationships.

Parity-Based Identities:
------------------------
- **BLACK_ONE (1)**: Identity for odd values, annihilator for even values
- **RED_ONE (-1)**: Identity for even values, annihilator for odd values

AVOS Product Parity Constraints:
---------------------------------
The AVOS (Algebraic Vertex-Ordered Semiring) product (⊗) enforces strict parity rules:

**RED_ONE (-1) Behavior:**
  - even ⊗ RED_ONE  = even  (acts as identity)
  - odd  ⊗ RED_ONE  = 0     (acts as annihilator/undefined)
  
  Examples:
    2 ⊗ RED_ONE = 2   (father remains father)
    4 ⊗ RED_ONE = 4   (paternal grandfather remains same)
    3 ⊗ RED_ONE = 0   (mother becomes undefined - wrong gender line)
    5 ⊗ RED_ONE = 0   (maternal grandfather becomes undefined)

**BLACK_ONE (1) Behavior:**
  - odd  ⊗ BLACK_ONE = odd   (acts as identity)
  - even ⊗ BLACK_ONE = 0     (acts as annihilator/undefined)
  
  Examples:
    3 ⊗ BLACK_ONE = 3   (mother remains mother)
    5 ⊗ BLACK_ONE = 5   (maternal grandfather remains same)
    2 ⊗ BLACK_ONE = 0   (father becomes undefined - wrong gender line)
    4 ⊗ BLACK_ONE = 0   (paternal grandfather becomes undefined)

**Why This Matters:**
This constraint prevents mixing incompatible gender lines in genealogical relationships.
In a red-black graph representing family trees:
  - Even values represent paternal (male) lineage
  - Odd values represent maternal (female) lineage
  - RED_ONE enforces "stay on paternal line or undefined"
  - BLACK_ONE enforces "stay on maternal line or undefined"

**Asymmetric Semantics:**
The parity constraints are NOT commutative because identities serve two different roles:
  - LEFT identity: "which gender am I starting from?" (no filtering, just composition)
  - RIGHT identity: "does this ancestor have the specified gender?" (parity filter)

For detailed mathematical justification and examples, see:
notebooks/Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb

Graph Theory Interpretation:
----------------------------
In a red-black adjacency matrix:
- RED_ONE appears on diagonal for "red" vertices (conventionally male/even)
- BLACK_ONE appears on diagonal for "black" vertices (conventionally female/odd)
- This ensures self-loops preserve the gender/parity constraint

NumPy Compatibility:
-------------------
For unsigned integer types, RED_ONE is represented as the maximum value (e.g., 255 
for uint8), which maintains the correct parity (maxint is always odd in binary).
This allows the same bitwise parity checks (x & 1) to work correctly across both
signed and unsigned integer types.
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
