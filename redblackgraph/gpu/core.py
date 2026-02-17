"""
Core AVOS operations on GPU using CuPy element-wise kernels.

.. deprecated::
    This module uses float32 ElementwiseKernels which lose precision for
    large integers (float32 has only a 24-bit mantissa). Use the int32
    RawKernel implementations in ``avos_kernels`` instead::

        from redblackgraph.gpu.avos_kernels import avos_sum_gpu, avos_product_gpu
"""

import warnings

from ._cuda_utils import CUPY_AVAILABLE, check_cupy

if CUPY_AVAILABLE:
    import cupy as cp
else:
    cp = None


# CuPy ElementwiseKernel for avos_sum
# avos_sum(x, y) = min(x, y) where both are non-zero, else the non-zero value
_avos_sum_kernel = None
if CUPY_AVAILABLE:
    _avos_sum_kernel = cp.ElementwiseKernel(
        'float32 x, float32 y',
        'float32 z',
        '''
        // Cast to int for integer operations
        int x_int = (int)x;
        int y_int = (int)y;
        int result;

        if (x_int == 0) {
            result = y_int;
        } else if (y_int == 0) {
            result = x_int;
        } else {
            result = (x_int < y_int) ? x_int : y_int;
        }

        // Cast back to float for CuPy sparse matrix compatibility
        z = (float)result;
        ''',
        'avos_sum'
    )


# CuPy ElementwiseKernel for avos_product with parity constraints
# This implements the full RED_ONE/BLACK_ONE semantics
_avos_product_kernel = None
if CUPY_AVAILABLE:
    _avos_product_kernel = cp.ElementwiseKernel(
        'float32 x, float32 y',
        'float32 z',
        '''
        // Cast to int for bitwise operations
        int x_int = (int)x;
        int y_int = (int)y;
        int result;

        // Negative values are invalid (aside from -1)
        if (x_int < -1 || y_int < -1) {
            result = 0;  // Error handling - in production would throw
            z = (float)result;
            return;
        }

        // Zero property
        if (x_int == 0 || y_int == 0) {
            result = 0;
            z = (float)result;
            return;
        }

        // Identity ⊗ Identity special cases
        if (x_int == -1 && y_int == -1) {
            result = -1;  // RED_ONE ⊗ RED_ONE = RED_ONE
            z = (float)result;
            return;
        }
        if (x_int == 1 && y_int == 1) {
            result = 1;   // BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE
            z = (float)result;
            return;
        }

        // Cross-gender identity cases
        if (x_int == -1 && y_int == 1) {
            result = 0;  // RED_ONE ⊗ BLACK_ONE undefined
            z = (float)result;
            return;
        }
        if (x_int == 1 && y_int == -1) {
            result = 0;  // BLACK_ONE ⊗ RED_ONE undefined
            z = (float)result;
            return;
        }

        // Identity on LEFT: starting point marker
        int x_adj = x_int;
        if (x_int == -1) {
            x_adj = 1;
        }

        // Identity on RIGHT: gender/parity filter
        if (y_int == -1) {
            // RED_ONE filter: even values only
            result = (x_adj & 1) ? 0 : x_adj;
            z = (float)result;
            return;
        }

        if (y_int == 1) {
            // BLACK_ONE filter: odd values only
            result = (x_adj & 1) ? x_adj : 0;
            z = (float)result;
            return;
        }

        // Normal case: bit shifting operation
        // Find MSB position of y
        int bit_position = 0;
        int y_temp = y_int;
        while (y_temp > 1) {
            y_temp >>= 1;
            bit_position++;
        }

        // Compute: (y & (2^bit_position - 1)) | (x << bit_position)
        int mask = (1 << bit_position) - 1;
        result = (y_int & mask) | (x_adj << bit_position);

        // Cast back to float for CuPy sparse matrix compatibility
        z = (float)result;
        ''',
        'avos_product'
    )


def avos_sum_gpu(x, y):
    """GPU version of avos_sum using CuPy.

    .. deprecated::
        Uses float32 which loses precision for large integers.
        Use ``redblackgraph.gpu.avos_kernels.avos_sum_gpu`` instead.
    """
    warnings.warn(
        "core.avos_sum_gpu uses float32 and loses precision for large integers. "
        "Use redblackgraph.gpu.avos_kernels.avos_sum_gpu (int32) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    check_cupy()
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    return _avos_sum_kernel(x, y)


def avos_product_gpu(x, y):
    """GPU version of avos_product with parity constraints using CuPy.

    .. deprecated::
        Uses float32 which loses precision for large integers.
        Use ``redblackgraph.gpu.avos_kernels.avos_product_gpu`` instead.
    """
    warnings.warn(
        "core.avos_product_gpu uses float32 and loses precision for large integers. "
        "Use redblackgraph.gpu.avos_kernels.avos_product_gpu (int32) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    check_cupy()
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    return _avos_product_kernel(x, y)
