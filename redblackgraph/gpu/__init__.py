"""
GPU-accelerated AVOS operations using CuPy and CUDA.

This is a minimal naive implementation to understand:
1. Build system integration with meson
2. CuPy integration for GPU arrays
3. CUDA kernel compilation
4. Installation on DGX Spark (Grace Hopper)

Status: Proof of concept / learning implementation
"""

__all__ = ['rb_matrix_gpu', 'avos_sum_gpu', 'avos_product_gpu']

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .core import avos_sum_gpu, avos_product_gpu
from .matrix import rb_matrix_gpu
