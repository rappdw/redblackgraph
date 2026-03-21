"""
GPU-accelerated AVOS operations using CuPy and CUDA.

Provides sparse matrix multiplication (SpGEMM) and transitive closure
on GPU using the AVOS semiring, with global memory hash tables for
unlimited output columns per row.

Production API:
- CSRMatrixGPU: Sparse matrix on GPU with raw int32 buffers
- spgemm: General A @ B sparse matrix multiply
- transitive_closure_gpu: Repeated squaring on GPU
"""

__all__ = [
    'CSRMatrixGPU',
    'spgemm', 'spgemm_upper_triangular', 'matmul_gpu',
    'transitive_closure_gpu',
    # Legacy (naive implementation)
    'rb_matrix_gpu', 'avos_sum_gpu', 'avos_product_gpu',
]

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


if CUPY_AVAILABLE:
    _try_preload_nvrtc()  # Best-effort: helps when using pip-installed CUDA wheels
    # Verify NVRTC actually works (system CUDA toolkit or preloaded wheels)
    try:
        cp.RawKernel('extern "C" __global__ void _rbg_probe() {}', '_rbg_probe')
    except Exception:
        CUPY_AVAILABLE = False
        cp = None

# Production API
from .csr_gpu import CSRMatrixGPU
from .spgemm import spgemm, spgemm_upper_triangular, matmul_gpu
from .transitive_closure import transitive_closure_gpu

# Legacy (naive implementation)
from .core import avos_sum_gpu, avos_product_gpu
from .matrix import rb_matrix_gpu
