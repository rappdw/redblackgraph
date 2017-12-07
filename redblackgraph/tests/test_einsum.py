import numpy as np
from redblackgraph import einsum

from numpy.testing import (
    run_module_suite, assert_, assert_equal, assert_array_equal,
    assert_almost_equal, assert_raises, suppress_warnings
    )

def test_avos():
    # test simple avos matmul
    A = np.array([[-1, 2, 3, 0, 0],
                  [0, -1, 0, 2, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0],
                  [2, 0, 0, 0, 1]], dtype=np.int64)
    S = np.array([[-1, 2, 3, 4, 0],
                  [0, -1, 0, 2, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0],
                  [2, 4, 5, 0, 1]], dtype=np.int64)
    assert_equal(einsum('ij, jk', A, A, avos=True), S)

    # test vector mat mul
    A_star = np.array([[-1, 2, 3, 4, 0],
                       [0, -1, 0, 2, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, -1, 0],
                       [2, 4, 5, 8, 1]], dtype=np.int64)
    u = np.array([2, 0, 0, 0, 0], dtype=np.int64)
    v = np.array([0, 3, 0, 0, 0], dtype=np.int64)
    u_lambda = np.array([2, 4, 5, 8, 0])
    v_lambda = np.array([5, 3, 0, 0, 9])
    assert_equal(einsum('ij,jk', u.reshape(1, 5), A_star, avos=True), u_lambda.reshape(1, 5))
    assert_equal(einsum('ij,jk', A_star, v.reshape(5, 1), avos=True), v_lambda.reshape(5, 1))
    assert_equal(einsum('i,...ij', u, A_star, avos=True), u_lambda)
    assert_equal(einsum('...i,i', A_star, v, avos=True), v_lambda)
