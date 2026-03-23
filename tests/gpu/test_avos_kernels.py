"""
Tests for production AVOS GPU kernels with raw int32 operations.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    from redblackgraph.gpu.avos_kernels import (
        avos_sum_gpu,
        avos_product_gpu,
        AVOS_SEMIRING,
        get_avos_kernels
    )
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")

# Import CPU reference for validation
from redblackgraph.reference.rbg_math import avos_sum as avos_sum_cpu
from redblackgraph.reference.rbg_math import avos_product as avos_product_cpu
from redblackgraph.constants import RED_ONE, BLACK_ONE


class TestAVOSSemiringSpec:
    """Test AVOS semiring specification."""
    
    def test_semiring_properties(self):
        """Verify AVOS semiring properties are correctly specified."""
        assert AVOS_SEMIRING.add == "avos_sum"
        assert AVOS_SEMIRING.mul == "avos_product"
        assert AVOS_SEMIRING.add_identity == "0"
        assert AVOS_SEMIRING.annihilator == "0"
        
        # Verify documented properties
        assert AVOS_SEMIRING.add_associative is True
        assert AVOS_SEMIRING.add_commutative is True
        assert AVOS_SEMIRING.mul_associative is False  # AVOS product NOT associative
        assert AVOS_SEMIRING.mul_commutative is False  # AVOS product NOT commutative


class TestAVOSSumKernel:
    """Test AVOS sum (non-zero minimum) kernel."""
    
    def test_sum_basic(self):
        """Test basic AVOS sum operations."""
        x = cp.array([2, 3, 4, 0, 5], dtype=cp.int32)
        y = cp.array([3, 2, 4, 7, 0], dtype=cp.int32)
        
        result = avos_sum_gpu(x, y)
        
        expected = cp.array([2, 2, 4, 7, 5], dtype=cp.int32)
        assert cp.array_equal(result, expected)
    
    def test_sum_zero_identity(self):
        """Test that 0 is the additive identity."""
        x = cp.array([2, 0, 5], dtype=cp.int32)
        y = cp.array([0, 3, 0], dtype=cp.int32)
        
        result = avos_sum_gpu(x, y)
        
        expected = cp.array([2, 3, 5], dtype=cp.int32)
        assert cp.array_equal(result, expected)
    
    def test_sum_both_zero(self):
        """Test 0 + 0 = 0."""
        x = cp.array([0, 0], dtype=cp.int32)
        y = cp.array([0, 0], dtype=cp.int32)
        
        result = avos_sum_gpu(x, y)
        
        expected = cp.array([0, 0], dtype=cp.int32)
        assert cp.array_equal(result, expected)
    
    def test_sum_vs_cpu_reference(self):
        """Validate GPU results against CPU reference."""
        test_values = [0, 1, -1, 2, 3, 5, 7, 10, 15, 20]
        
        for x_val in test_values:
            for y_val in test_values:
                x_gpu = cp.array([x_val], dtype=cp.int32)
                y_gpu = cp.array([y_val], dtype=cp.int32)
                
                result_gpu = avos_sum_gpu(x_gpu, y_gpu).get()[0]
                result_cpu = avos_sum_cpu(x_val, y_val)
                
                assert result_gpu == result_cpu, f"Mismatch: sum({x_val}, {y_val})"


class TestAVOSProductKernel:
    """Test AVOS product kernel with parity constraints."""
    
    def test_product_zero_annihilator(self):
        """Test that 0 annihilates in multiplication."""
        x = cp.array([0, 5, 0], dtype=cp.int32)
        y = cp.array([3, 0, 0], dtype=cp.int32)
        
        result = avos_product_gpu(x, y)
        
        expected = cp.array([0, 0, 0], dtype=cp.int32)
        assert cp.array_equal(result, expected)
    
    def test_product_identities(self):
        """Test RED_ONE and BLACK_ONE identity behavior."""
        # RED_ONE ⊗ RED_ONE = RED_ONE
        result = avos_product_gpu(
            cp.array([RED_ONE], dtype=cp.int32),
            cp.array([RED_ONE], dtype=cp.int32)
        )
        assert result.get()[0] == RED_ONE
        
        # BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE
        result = avos_product_gpu(
            cp.array([BLACK_ONE], dtype=cp.int32),
            cp.array([BLACK_ONE], dtype=cp.int32)
        )
        assert result.get()[0] == BLACK_ONE
        
        # Cross-gender products = 0
        result = avos_product_gpu(
            cp.array([RED_ONE], dtype=cp.int32),
            cp.array([BLACK_ONE], dtype=cp.int32)
        )
        assert result.get()[0] == 0
        
        result = avos_product_gpu(
            cp.array([BLACK_ONE], dtype=cp.int32),
            cp.array([RED_ONE], dtype=cp.int32)
        )
        assert result.get()[0] == 0
    
    def test_product_parity_constraints(self):
        """Test parity filtering on right operand."""
        # even ⊗ RED_ONE = even (male identity)
        even_val = cp.array([4], dtype=cp.int32)
        result = avos_product_gpu(even_val, cp.array([RED_ONE], dtype=cp.int32))
        assert result.get()[0] == 4
        
        # odd ⊗ RED_ONE = 0 (no male self for female)
        odd_val = cp.array([5], dtype=cp.int32)
        result = avos_product_gpu(odd_val, cp.array([RED_ONE], dtype=cp.int32))
        assert result.get()[0] == 0
        
        # odd ⊗ BLACK_ONE = odd (female identity)
        odd_val = cp.array([5], dtype=cp.int32)
        result = avos_product_gpu(odd_val, cp.array([BLACK_ONE], dtype=cp.int32))
        assert result.get()[0] == 5
        
        # even ⊗ BLACK_ONE = 0 (no female self for male)
        even_val = cp.array([4], dtype=cp.int32)
        result = avos_product_gpu(even_val, cp.array([BLACK_ONE], dtype=cp.int32))
        assert result.get()[0] == 0
    
    def test_product_vs_cpu_reference(self):
        """Validate GPU results against CPU reference."""
        test_values = [0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
        
        mismatches = []
        for x_val in test_values:
            for y_val in test_values:
                x_gpu = cp.array([x_val], dtype=cp.int32)
                y_gpu = cp.array([y_val], dtype=cp.int32)
                
                result_gpu = avos_product_gpu(x_gpu, y_gpu).get()[0]
                result_cpu = avos_product_cpu(x_val, y_val)
                
                if result_gpu != result_cpu:
                    mismatches.append((x_val, y_val, result_cpu, result_gpu))
        
        if mismatches:
            msg = "\n".join(
                f"product({x}, {y}): CPU={cpu}, GPU={gpu}"
                for x, y, cpu, gpu in mismatches[:10]  # Show first 10
            )
            pytest.fail(f"GPU/CPU mismatch:\n{msg}")


class TestKernelVectorized:
    """Test vectorized operations on larger arrays."""
    
    def test_sum_large_array(self):
        """Test AVOS sum on large array."""
        n = 10000
        x = cp.random.randint(0, 100, size=n, dtype=cp.int32)
        y = cp.random.randint(0, 100, size=n, dtype=cp.int32)
        
        result = avos_sum_gpu(x, y)
        
        # Validate shape and dtype
        assert result.shape == (n,)
        assert result.dtype == cp.int32
        
        # Spot check a few values
        x_cpu = x.get()
        y_cpu = y.get()
        result_cpu_arr = result.get()
        
        for i in range(0, n, n//10):
            expected = avos_sum_cpu(int(x_cpu[i]), int(y_cpu[i]))
            assert result_cpu_arr[i] == expected
    
    def test_product_large_array(self):
        """Test AVOS product on large array."""
        n = 10000
        x = cp.random.randint(0, 50, size=n, dtype=cp.int32)
        y = cp.random.randint(0, 50, size=n, dtype=cp.int32)
        
        result = avos_product_gpu(x, y)
        
        # Validate shape and dtype
        assert result.shape == (n,)
        assert result.dtype == cp.int32


class TestKernelCaching:
    """Test kernel compilation and caching."""
    
    def test_singleton_kernels(self):
        """Test that kernels are compiled once and cached."""
        kernels1 = get_avos_kernels()
        kernels2 = get_avos_kernels()
        
        assert kernels1 is kernels2  # Same instance


class TestShapeMismatch:
    """Test error handling for shape mismatches."""
    
    def test_sum_shape_mismatch(self):
        """Test that shape mismatch raises error in sum."""
        x = cp.array([1, 2, 3], dtype=cp.int32)
        y = cp.array([1, 2], dtype=cp.int32)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            avos_sum_gpu(x, y)
    
    def test_product_shape_mismatch(self):
        """Test that shape mismatch raises error in product."""
        x = cp.array([1, 2, 3], dtype=cp.int32)
        y = cp.array([1, 2], dtype=cp.int32)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            avos_product_gpu(x, y)
