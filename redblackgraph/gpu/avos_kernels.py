"""
AVOS operations as CUDA kernels for GPU sparse matrix operations.

This module implements proper int32 AVOS operations using CuPy RawKernels
for use in SpGEMM numeric phase.

References:
- RED_ONE = -1 (male identity/filter)
- BLACK_ONE = 1 (female identity/filter)
- AVOS product has asymmetric identity behavior with parity constraints
"""

from typing import NamedTuple

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def _try_preload_nvrtc() -> bool:
    if not CUPY_AVAILABLE:
        return False

    try:
        import ctypes
        import glob
        import os
        import site

        for sp in site.getsitepackages():
            cand = glob.glob(os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"))
            if cand:
                cand.sort(reverse=True)
                ctypes.CDLL(cand[0], mode=ctypes.RTLD_GLOBAL)
                return True

        return False
    except Exception:
        return False


if CUPY_AVAILABLE and not _try_preload_nvrtc():
    # Without NVRTC, CuPy will raise at kernel compilation time.
    raise ImportError(
        "CuPy is installed but NVRTC (libnvrtc.so) could not be loaded. "
        "Install the CUDA NVRTC runtime (e.g. nvidia-cuda-nvrtc-cu12) "
        "or ensure libnvrtc.so is on your loader path."
    )


class SemiringSpec(NamedTuple):
    """
    Specification of algebraic semiring for matrix operations.
    
    Attributes:
        add: Name of device function for "addition" operation
        mul: Name of device function for "multiplication" operation
        add_identity: Literal value for additive identity
        mul_identity: Literal value for multiplicative identity
        annihilator: Literal value for multiplicative zero
        add_associative: Whether addition is associative
        add_commutative: Whether addition is commutative
        add_idempotent: Whether a + a = a
        mul_associative: Whether multiplication is associative
        mul_commutative: Whether multiplication is commutative
    """
    add: str
    mul: str
    add_identity: str
    mul_identity: str
    annihilator: str
    add_associative: bool
    add_commutative: bool
    add_idempotent: bool
    mul_associative: bool
    mul_commutative: bool


# AVOS Semiring specification
AVOS_SEMIRING = SemiringSpec(
    add="avos_sum",
    mul="avos_product",
    add_identity="0",
    mul_identity="1",  # Note: asymmetric - see parity constraints
    annihilator="0",
    add_associative=True,
    add_commutative=True,
    add_idempotent=False,  # min(x,x) = x when both non-zero, but 0 is special
    mul_associative=False,  # AVOS product is NOT associative
    mul_commutative=False   # AVOS product is NOT commutative
)


# CUDA device code for AVOS operations
AVOS_DEVICE_CODE = r'''
extern "C" {

// AVOS sum: Non-zero minimum
// If either operand is 0 (neutral), return the other
// Otherwise return the minimum (non-zero values only)
__device__ inline int avos_sum(int x, int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    return (x < y) ? x : y;
}

// Helper function: Find MSB position (most significant bit)
__device__ inline int MSB(int x) {
    int bit_position = 0;
    while (x > 1) {
        x >>= 1;
        bit_position++;
    }
    return bit_position;
}

// AVOS product: Bitwise operation with parity constraints
// Implements asymmetric identity behavior:
// - LEFT identity: treated as 1 for composition
// - RIGHT identity: parity-based filtering
//
// Special cases:
// - RED_ONE (-1) and BLACK_ONE (1) have special semantics
// - Cross-gender products (-1 ⊗ 1) and (1 ⊗ -1) return 0
//
// See MATHEMATICAL_ANALYSIS.md for complete semantics
__device__ inline int avos_product(int x, int y) {
    // Zero annihilates
    if (x == 0 || y == 0) return 0;
    
    // Identity cases
    const int RED_ONE = -1;
    const int BLACK_ONE = 1;
    
    // Identity ⊗ Identity special cases (must come before other checks)
    if (x == RED_ONE && y == RED_ONE) {
        return RED_ONE;  // RED_ONE ⊗ RED_ONE = RED_ONE (male self-loop)
    }
    if (x == BLACK_ONE && y == BLACK_ONE) {
        return BLACK_ONE;  // BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE (female self-loop)
    }
    if (x == RED_ONE && y == BLACK_ONE) {
        return 0;  // RED_ONE ⊗ BLACK_ONE: male's female self is undefined
    }
    if (x == BLACK_ONE && y == RED_ONE) {
        return 0;  // BLACK_ONE ⊗ RED_ONE: female's male self is undefined
    }
    
    // LEFT identity (x): starting point marker, treat as 1 for composition
    if (x == RED_ONE) {
        x = 1;  // Treat RED_ONE as 1 for bit-shift operation
    }
    
    // RIGHT identity (y): gender filter with parity constraints
    if (y == RED_ONE) {
        // value ⊗ RED_ONE: return value if even (male), 0 if odd (female)
        return (x & 1) ? 0 : x;
    }
    if (y == BLACK_ONE) {
        // value ⊗ BLACK_ONE: return value if odd (female), 0 if even (male)
        return (x & 1) ? x : 0;
    }
    
    // General case: bit shifting composition
    // Formula: ((y & (2^bit_position - 1)) | (x << bit_position))
    int bit_position = MSB(y);
    int mask = (1 << bit_position) - 1;  // 2^bit_position - 1
    int result = (y & mask) | (x << bit_position);
    
    return result;
}

// Element-wise AVOS sum kernel
__global__ void avos_sum_kernel(
    const int* __restrict__ x,
    const int* __restrict__ y,
    int* __restrict__ z,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = avos_sum(x[i], y[i]);
    }
}

// Element-wise AVOS product kernel
__global__ void avos_product_kernel(
    const int* __restrict__ x,
    const int* __restrict__ y,
    int* __restrict__ z,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = avos_product(x[i], y[i]);
    }
}

} // extern "C"
'''


class AVOSKernels:
    """
    Compiled CUDA kernels for AVOS operations.
    
    Uses CuPy RawKernel for direct CUDA compilation and optimal performance.
    Kernels are compiled once and cached.
    """
    
    def __init__(self):
        """Initialize and compile AVOS kernels."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")
        
        # Compile kernels
        self._module = cp.RawModule(code=AVOS_DEVICE_CODE)
        self._sum_kernel = self._module.get_function('avos_sum_kernel')
        self._product_kernel = self._module.get_function('avos_product_kernel')
    
    def avos_sum(self, x: 'cp.ndarray', y: 'cp.ndarray') -> 'cp.ndarray':
        """
        Element-wise AVOS sum (non-zero minimum).
        
        Args:
            x: First array (int32)
            y: Second array (int32)
        
        Returns:
            Result array (int32)
        """
        x = cp.asarray(x, dtype=cp.int32)
        y = cp.asarray(y, dtype=cp.int32)
        
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        z = cp.empty_like(x)
        n = x.size
        
        # Launch kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        self._sum_kernel(
            (grid_size,), (block_size,),
            (x, y, z, n)
        )
        
        return z
    
    def avos_product(self, x: 'cp.ndarray', y: 'cp.ndarray') -> 'cp.ndarray':
        """
        Element-wise AVOS product with parity constraints.
        
        Args:
            x: First array (int32)
            y: Second array (int32)
        
        Returns:
            Result array (int32)
        """
        x = cp.asarray(x, dtype=cp.int32)
        y = cp.asarray(y, dtype=cp.int32)
        
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        z = cp.empty_like(x)
        n = x.size
        
        # Launch kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        self._product_kernel(
            (grid_size,), (block_size,),
            (x, y, z, n)
        )
        
        return z


# Singleton instance
_kernels = None


def get_avos_kernels() -> AVOSKernels:
    """Get or create singleton AVOS kernels instance."""
    global _kernels
    if _kernels is None:
        _kernels = AVOSKernels()
    return _kernels


def avos_sum_gpu(x: 'cp.ndarray', y: 'cp.ndarray') -> 'cp.ndarray':
    """
    Element-wise AVOS sum (non-zero minimum).
    
    Args:
        x: First array (int32)
        y: Second array (int32)
    
    Returns:
        Result array (int32)
    """
    return get_avos_kernels().avos_sum(x, y)


def avos_product_gpu(x: 'cp.ndarray', y: 'cp.ndarray') -> 'cp.ndarray':
    """
    Element-wise AVOS product with parity constraints.
    
    Args:
        x: First array (int32)
        y: Second array (int32)
    
    Returns:
        Result array (int32)
    """
    return get_avos_kernels().avos_product(x, y)
