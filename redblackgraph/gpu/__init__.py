"""
GPU-accelerated Red-Black Graph operations using CuPy and CUDA.

Exports
-------
Availability:
    CUPY_AVAILABLE       — True when CuPy + CUDA GPU are usable
    is_gpu_available()   — same, as a callable

Element-wise AVOS ops (production int32 via RawKernel):
    avos_sum_gpu, avos_product_gpu

Sparse matrix types and SpGEMM:
    CSRMatrixGPU
    spgemm_upper_triangular, matmul_gpu, spgemm_with_stats, SpGEMMStats

High-level matrix wrapper:
    rb_matrix_gpu

Device policy:
    DevicePolicy, get_device_policy, set_device_policy, device, resolve_device
"""

# Single source of truth for CuPy / CUDA availability
from ._cuda_utils import CUPY_AVAILABLE, check_cupy, is_gpu_available

# Device policy (works without CuPy)
from ._device_policy import (
    DevicePolicy,
    device,
    get_device_policy,
    resolve_device,
    set_device_policy,
)

__all__ = [
    # Availability
    "CUPY_AVAILABLE",
    "is_gpu_available",
    # Device policy
    "DevicePolicy",
    "device",
    "get_device_policy",
    "set_device_policy",
    "resolve_device",
]

# Production int32 AVOS ops and sparse infrastructure require CuPy at import
# time because the CUDA kernels are compiled eagerly.  Guard so that
# ``import redblackgraph.gpu`` never fails on CPU-only machines.
if CUPY_AVAILABLE:
    # Production element-wise AVOS operations (int32 RawKernel)
    from .avos_kernels import avos_product_gpu, avos_sum_gpu

    # CSR matrix and SpGEMM
    from .csr_gpu import CSRMatrixGPU
    from .spgemm import (
        SpGEMMStats,
        matmul_gpu,
        spgemm_upper_triangular,
        spgemm_with_stats,
    )

    # High-level matrix wrapper
    from .matrix import rb_matrix_gpu

    __all__ += [
        "avos_sum_gpu",
        "avos_product_gpu",
        "CSRMatrixGPU",
        "spgemm_upper_triangular",
        "matmul_gpu",
        "spgemm_with_stats",
        "SpGEMMStats",
        "rb_matrix_gpu",
    ]
