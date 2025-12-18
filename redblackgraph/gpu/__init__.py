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


def _try_preload_nvrtc() -> bool:
    if not CUPY_AVAILABLE:
        return False

    try:
        import ctypes
        import glob
        import os
        import site

        # CuPy loads several CUDA libraries via dlopen("lib*.so.*"). If the user
        # installed NVIDIA Python runtime wheels, preload those libraries from
        # site-packages so they are discoverable even without a system CUDA toolkit.
        loaded_any = False
        for sp in site.getsitepackages():
            patterns = [
                os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"),
                os.path.join(sp, "nvidia", "cublas", "lib", "libcublas.so*"),
                os.path.join(sp, "nvidia", "cusparse", "lib", "libcusparse.so*"),
            ]

            for pat in patterns:
                cand = glob.glob(pat)
                if cand:
                    cand.sort(reverse=True)
                    ctypes.CDLL(cand[0], mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True

        return loaded_any
    except Exception:
        return False


if CUPY_AVAILABLE and not _try_preload_nvrtc():
    # If NVRTC can't be loaded, CuPy kernels will fail at runtime.
    # Mark GPU as unavailable so tests can skip cleanly.
    CUPY_AVAILABLE = False
    cp = None

from .core import avos_sum_gpu, avos_product_gpu
from .matrix import rb_matrix_gpu
