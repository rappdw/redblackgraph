"""
Single source of truth for CUDA/CuPy availability and initialization.

All GPU modules should import CUPY_AVAILABLE and check_cupy from here
instead of maintaining their own copies.
"""

import ctypes
import glob
import os
import site

try:
    import cupy as cp
    _CUPY_IMPORT_OK = True
except ImportError:
    _CUPY_IMPORT_OK = False
    cp = None


def _try_preload_nvrtc() -> bool:
    """Preload NVIDIA runtime libraries from pip-installed wheels.

    CuPy loads CUDA libraries via dlopen. When NVIDIA Python runtime wheels
    are installed (e.g. nvidia-cuda-nvrtc-cu12), the shared objects live
    under site-packages and may not be on the default loader path.
    This function finds and preloads them so CuPy can compile kernels.

    Returns:
        True if at least one library was preloaded.
    """
    if not _CUPY_IMPORT_OK:
        return False

    try:
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


# Determine overall CuPy availability.
# If CuPy imported but NVRTC cannot be loaded, kernels will fail at runtime,
# so we treat that as "unavailable".
if _CUPY_IMPORT_OK:
    _try_preload_nvrtc()
    # Verify we can actually use CuPy by checking device count
    try:
        _device_count = cp.cuda.runtime.getDeviceCount()
        CUPY_AVAILABLE = _device_count > 0
    except Exception:
        # NVRTC preload failed and no system CUDA — mark unavailable
        CUPY_AVAILABLE = False
else:
    CUPY_AVAILABLE = False


def check_cupy():
    """Raise ImportError if CuPy/CUDA is not available."""
    if not CUPY_AVAILABLE:
        raise ImportError(
            "CuPy with a CUDA GPU is required for GPU operations. "
            "Install with: pip install cupy-cuda12x (or appropriate CUDA version)"
        )


def is_gpu_available() -> bool:
    """Check whether GPU acceleration is available.

    Returns True when CuPy is installed, NVRTC is loadable, and at least
    one CUDA device is present.
    """
    return CUPY_AVAILABLE
