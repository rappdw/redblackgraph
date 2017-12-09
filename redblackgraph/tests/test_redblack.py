import numpy as np
import redblackgraph as rbg
from numpy.testing import (
    run_module_suite, assert_equal
    )


class TestmatrixOperations(object):

    def test_avos(self):
        # test simple avos matmul
        A = rbg.array([[-1,  2,  3,  0,  0],
                         [ 0, -1,  0,  2,  0],
                         [ 0,  0,  1,  0,  0],
                         [ 0,  0,  0,  -1, 0],
                         [ 2,  0,  0,  0,  1]], dtype=np.int64)
        S = rbg.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)
        A = rbg.matrix([[-1,  2,  3,  0,  0],
                          [ 0, -1,  0,  2,  0],
                          [ 0,  0,  1,  0,  0],
                          [ 0,  0,  0,  -1, 0],
                          [ 2,  0,  0,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)

        A_star = rbg.array([[-1,  2,  3,  4,  0],
                              [ 0, -1,  0,  2,  0],
                              [ 0,  0,  1,  0,  0],
                              [ 0,  0,  0, -1,  0],
                              [ 2,  4,  5,  8,  1]], dtype=np.int64)
        assert_equal(S @ A, A_star)
        assert_equal(A @ (A @ A), A_star)
        assert_equal(A @ A @ A, A_star)
        assert_equal((A @ A) @ A, A_star)

        # test vector mat mul

        # using rank 1 arrays can cause problems.
        # See: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/87MUx/a-note-on-python-numpy-vectors
        # Safer to always use either a row vector or column vector
        u = rbg.array([2, 0, 0, 0, 0], dtype=np.int64).reshape((1, 5))
        v = rbg.array([0, 3, 0, 0, 0], dtype=np.int64).reshape((5, 1))
        u_lambda = np.array([2, 4, 5, 8, 0]).reshape((1, 5))
        v_lambda = np.array([5, 3, 0, 0, 9]).reshape((5, 1))
        assert_equal(u @ A_star, u_lambda)
        assert_equal(A_star @ v, v_lambda)
        A_star = rbg.matrix([[-1,  2,  3,  4,  0],
                               [ 0, -1,  0,  2,  0],
                               [ 0,  0,  1,  0,  0],
                               [ 0,  0,  0, -1,  0],
                               [ 2,  4,  5,  8,  1]], dtype=np.int64)
        bar = u @ A_star
        assert_equal(bar, u_lambda)
        assert_equal(A_star @ v, v_lambda)
        assert_equal(A @ A @ A @ v, v_lambda)


if __name__ == "__main__":
    run_module_suite()
