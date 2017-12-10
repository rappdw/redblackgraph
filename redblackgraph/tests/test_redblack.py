import numpy as np
import redblackgraph as rb
from numpy.testing import (
    run_module_suite, assert_equal
    )


class TestmatrixOperations(object):

    def test_avos(self):
        # test simple avos matmul
        A = rb.array([[-1,  2,  3,  0,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0,  -1, 0],
                      [ 2,  0,  0,  0,  1]], dtype=np.int64)
        S = rb.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)
        A = rb.matrix([[-1,  2,  3,  0,  0],
                       [ 0, -1,  0,  2,  0],
                       [ 0,  0,  1,  0,  0],
                       [ 0,  0,  0,  -1, 0],
                       [ 2,  0,  0,  0,  1]], dtype=np.int64)
        assert_equal(A @ A, S)

        A_star = rb.array([[-1,  2,  3,  4,  0],
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
        u = rb.array([2, 0, 0, 0, 0], dtype=np.int64).reshape((1, 5))
        v = rb.array([0, 3, 0, 0, 0], dtype=np.int64).reshape((5, 1))
        u_lambda = np.array([2, 4, 5, 8, 0]).reshape((1, 5))
        v_lambda = np.array([5, 3, 0, 0, 9]).reshape((5, 1))
        assert_equal(u @ A_star, u_lambda)
        assert_equal(A_star @ v, v_lambda)
        A_star = rb.matrix([[-1,  2,  3,  4,  0],
                            [ 0, -1,  0,  2,  0],
                            [ 0,  0,  1,  0,  0],
                            [ 0,  0,  0, -1,  0],
                            [ 2,  4,  5,  8,  1]], dtype=np.int64)
        bar = u @ A_star
        assert_equal(bar, u_lambda)
        assert_equal(A_star @ v, v_lambda)
        assert_equal(A @ A @ A @ v, v_lambda)

    def test_relational_composition(self):
        A = rb.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  8,  1]])
        u = rb.array([[2, 0, 0, 0, 0]])
        v = rb.array([[0],
                      [0],
                      [0],
                      [0],
                      [0]])
        A_lambda = A.relational_composition(u, v, -1)
        assert A_lambda is not None

    def test_warshall(self):
        a = rb.array([[-1,  2,  3,  0,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  0,  0,  0,  1]])
        expected = rb.array([[-1,  2,  3,  4,  0],
                             [ 0, -1,  0,  2,  0],
                             [ 0,  0,  1,  0,  0],
                             [ 0,  0,  0, -1,  0],
                             [ 2,  4,  5,  8,  1]])
        results = rb.warshall(a)
        assert_equal(results[0], expected)
        assert_equal(results[1], 3)


if __name__ == "__main__":
    run_module_suite()
