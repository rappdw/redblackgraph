import numpy as np
import redblackgraph as rbg
from numpy.testing import (
    run_module_suite, assert_equal
    )


class TestRbMatrixOperations(object):

    def test_avos(self):
        # test simple avos matmul
        A = rbg.rbarray([[-1,  2,  3,  0,  0],
                         [ 0, -1,  0,  2,  0],
                         [ 0,  0,  1,  0,  0],
                         [ 0,  0,  0,  -1, 0],
                         [ 2,  0,  0,  0,  1]], dtype=np.int64)
        S = np.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)
        A = rbg.rbmatrix([[-1,  2,  3,  0,  0],
                          [ 0, -1,  0,  2,  0],
                          [ 0,  0,  1,  0,  0],
                          [ 0,  0,  0,  -1, 0],
                          [ 2,  0,  0,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)

        # test vector mat mul
        A_star = rbg.rbarray([[-1,  2,  3,  4,  0],
                              [ 0, -1,  0,  2,  0],
                              [ 0,  0,  1,  0,  0],
                              [ 0,  0,  0, -1,  0],
                              [ 2,  4,  5,  8,  1]], dtype=np.int64)
        u = rbg.rbarray([2, 0, 0, 0, 0], dtype=np.int64)
        v = rbg.rbarray([0, 3, 0, 0, 0], dtype=np.int64)
        u_lambda = np.array([2, 4, 5, 8, 0])
        v_lambda = np.array([5, 3, 0, 0, 9])
        assert_equal(u @ A_star, u_lambda)
        assert_equal(A_star @ v, v_lambda)
        A_star = rbg.rbmatrix([[-1,  2,  3,  4,  0],
                               [ 0, -1,  0,  2,  0],
                               [ 0,  0,  1,  0,  0],
                               [ 0,  0,  0, -1,  0],
                               [ 2,  4,  5,  8,  1]], dtype=np.int64)
        assert_equal(u @ A_star, u_lambda)
        assert_equal(A_star @ v, v_lambda)

if __name__ == "__main__":
    run_module_suite()
