import numpy as np
import pytest
from numpy.testing import assert_equal

import redblackgraph as rb


@pytest.mark.parametrize("dtype", [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64
])
def test_warshall(dtype):
    a = rb.array([[-1,  2,  3,  0,  0],
                  [ 0, -1,  0,  2,  0],
                  [ 0,  0,  1,  0,  0],
                  [ 0,  0,  0, -1,  0],
                  [ 2,  0,  0,  0,  1]], dtype=dtype)
    expected = rb.array([[-1,  2,  3,  4,  0],
                         [ 0, -1,  0,  2,  0],
                         [ 0,  0,  1,  0,  0],
                         [ 0,  0,  0, -1,  0],
                         [ 2,  4,  5,  8,  1]], dtype=dtype)
    results = a.transitive_closure()
    assert_equal(results.W, expected)
    assert_equal(results.diameter, 3)