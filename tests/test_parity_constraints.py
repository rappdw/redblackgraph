"""
Test parity identity constraints for RED_ONE and BLACK_ONE in AVOS product.

The red-black semiring has two multiplicative identities based on parity:
- RED_ONE (-1): Identity for even values, annihilator for odd values
- BLACK_ONE (1): Identity for odd values, annihilator for even values
"""
import numpy as np
import pytest
from redblackgraph import RED_ONE, BLACK_ONE, avos_product
from numpy.testing import assert_equal


@pytest.mark.parametrize("dtype", [
    np.int8, np.uint8,
    np.int16, np.uint16,
    np.int32, np.uint32,
    np.int64, np.uint64
])
class TestParityConstraints:
    """Test the parity identity constraints of the red-black semiring."""
    
    def test_red_one_identity_for_even(self, dtype):
        """RED_ONE acts as identity for even values."""
        # Test various even values
        even_values = [2, 4, 6, 8, 10, 100, 256, 1000]
        
        for val in even_values:
            if val < np.iinfo(dtype).max:
                # even ⊗ RED_ONE = even
                result = avos_product(val, RED_ONE)
                assert result == val, f"{val} ⊗ RED_ONE should equal {val}, got {result}"
                
                # RED_ONE ⊗ even = even
                result = avos_product(RED_ONE, val)
                assert result == val, f"RED_ONE ⊗ {val} should equal {val}, got {result}"
    
    def test_red_one_annihilates_odd(self, dtype):
        """RED_ONE annihilates odd values when on the RIGHT (gender filter)."""
        # Test various odd values
        odd_values = [3, 5, 7, 9, 11, 99, 255, 999]
        
        for val in odd_values:
            if val < np.iinfo(dtype).max:
                # odd ⊗ RED_ONE = 0 (odd values have no male self)
                result = avos_product(val, RED_ONE)
                assert result == 0, f"{val} ⊗ RED_ONE should equal 0, got {result}"
                
                # RED_ONE ⊗ odd = odd (male-me's odd-ancestor)
                result = avos_product(RED_ONE, val)
                assert result == val, f"RED_ONE ⊗ {val} should equal {val}, got {result}"
    
    def test_black_one_identity_for_odd(self, dtype):
        """BLACK_ONE acts as identity for odd values."""
        # Test various odd values
        odd_values = [3, 5, 7, 9, 11, 99, 255, 999]
        
        for val in odd_values:
            if val < np.iinfo(dtype).max:
                # odd ⊗ BLACK_ONE = odd
                result = avos_product(val, BLACK_ONE)
                assert result == val, f"{val} ⊗ BLACK_ONE should equal {val}, got {result}"
                
                # BLACK_ONE ⊗ odd = odd  
                result = avos_product(BLACK_ONE, val)
                assert result == val, f"BLACK_ONE ⊗ {val} should equal {val}, got {result}"
    
    def test_black_one_annihilates_even(self, dtype):
        """BLACK_ONE annihilates even values when on the RIGHT (gender filter)."""
        # Test various even values
        even_values = [2, 4, 6, 8, 10, 100, 256, 1000]
        
        for val in even_values:
            if val < np.iinfo(dtype).max:
                # even ⊗ BLACK_ONE = 0 (even values have no female self)
                result = avos_product(val, BLACK_ONE)
                assert result == 0, f"{val} ⊗ BLACK_ONE should equal 0, got {result}"
                
                # BLACK_ONE ⊗ even = even (female-me's even-ancestor)
                result = avos_product(BLACK_ONE, val)
                assert result == val, f"BLACK_ONE ⊗ {val} should equal {val}, got {result}"
    
    def test_special_case_black_one_with_black_one(self, dtype):
        """BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE (1 is odd, so BLACK_ONE is its identity)."""
        result = avos_product(BLACK_ONE, BLACK_ONE)
        assert result == BLACK_ONE, f"BLACK_ONE ⊗ BLACK_ONE should equal BLACK_ONE, got {result}"
    
    def test_special_case_red_one_with_red_one(self, dtype):
        """RED_ONE ⊗ RED_ONE = RED_ONE (-1 is even, so RED_ONE is its identity)."""
        result = avos_product(RED_ONE, RED_ONE)
        assert result == RED_ONE, f"RED_ONE ⊗ RED_ONE should equal RED_ONE, got {result}"


def test_parity_examples():
    """Document specific examples of parity constraints with clear semantics."""
    
    # RED_ONE with even values (identity)
    assert avos_product(2, RED_ONE) == 2, "2 (father) ⊗ RED_ONE = 2"
    assert avos_product(4, RED_ONE) == 4, "4 (paternal grandfather) ⊗ RED_ONE = 4"
    assert avos_product(6, RED_ONE) == 6, "6 (paternal great-grandfather) ⊗ RED_ONE = 6"
    
    # RED_ONE with odd values (annihilator)
    assert avos_product(3, RED_ONE) == 0, "3 (mother) ⊗ RED_ONE = 0 (undefined)"
    assert avos_product(5, RED_ONE) == 0, "5 (maternal grandfather) ⊗ RED_ONE = 0 (undefined)"
    assert avos_product(7, RED_ONE) == 0, "7 (maternal great-grandmother) ⊗ RED_ONE = 0 (undefined)"
    
    # BLACK_ONE with odd values (identity)
    assert avos_product(3, BLACK_ONE) == 3, "3 (mother) ⊗ BLACK_ONE = 3"
    assert avos_product(5, BLACK_ONE) == 5, "5 (maternal grandfather) ⊗ BLACK_ONE = 5"
    assert avos_product(7, BLACK_ONE) == 7, "7 (maternal great-grandmother) ⊗ BLACK_ONE = 7"
    
    # BLACK_ONE with even values (annihilator)
    assert avos_product(2, BLACK_ONE) == 0, "2 (father) ⊗ BLACK_ONE = 0 (undefined)"
    assert avos_product(4, BLACK_ONE) == 0, "4 (paternal grandfather) ⊗ BLACK_ONE = 0 (undefined)"
    assert avos_product(6, BLACK_ONE) == 0, "6 (paternal great-grandfather) ⊗ BLACK_ONE = 0 (undefined)"


def test_asymmetry():
    """Test that identities are ASYMMETRIC: left is starting point, right is filter."""
    
    # RED_ONE tests (even identity/filter)
    # Left side: RED_ONE is just "male-me" starting point
    assert avos_product(RED_ONE, 3) == 3, "RED_ONE ⊗ 3 = male-me's mother = 3"
    assert avos_product(RED_ONE, 2) == 2, "RED_ONE ⊗ 2 = male-me's father = 2"
    
    # Right side: RED_ONE is male/even filter
    assert avos_product(3, RED_ONE) == 0, "3 ⊗ RED_ONE = mother's male self = 0 (undefined)"
    assert avos_product(2, RED_ONE) == 2, "2 ⊗ RED_ONE = father's male self = 2"
    
    # BLACK_ONE tests (odd identity/filter)
    # Left side: BLACK_ONE is just "female-me" starting point
    assert avos_product(BLACK_ONE, 3) == 3, "BLACK_ONE ⊗ 3 = female-me's mother = 3"
    assert avos_product(BLACK_ONE, 2) == 2, "BLACK_ONE ⊗ 2 = female-me's father = 2"
    
    # Right side: BLACK_ONE is female/odd filter
    assert avos_product(3, BLACK_ONE) == 3, "3 ⊗ BLACK_ONE = mother's female self = 3"
    assert avos_product(2, BLACK_ONE) == 0, "2 ⊗ BLACK_ONE = father's female self = 0 (undefined)"
    
    # Cross-gender identity cases (both should be undefined due to parity)
    assert avos_product(RED_ONE, BLACK_ONE) == 0, "RED_ONE ⊗ BLACK_ONE = male's female self = 0"
    assert avos_product(BLACK_ONE, RED_ONE) == 0, "BLACK_ONE ⊗ RED_ONE = female's male self = 0"
