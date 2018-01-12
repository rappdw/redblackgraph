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
                      [0],
                      ])
        A_lambda = A.vertex_relational_composition(u, v, -1)
        expected = rb.array([[-1,  2,  3,  4,  0,  0],
                             [ 0, -1,  0,  2,  0,  0],
                             [ 0,  0,  1,  0,  0,  0],
                             [ 0,  0,  0, -1,  0,  0],
                             [ 2,  4,  5,  8,  1,  0],
                             [ 2,  4,  5,  8,  0, -1]])
        assert_equal(A_lambda, expected)

        u = rb.array([[0, 0, 0, 0, 0, 0]])
        v = rb.array([[0],
                      [3],
                      [0],
                      [0],
                      [0],
                      [0],
                      ])
        A_lambda_2 = A_lambda.vertex_relational_composition(u, v, 1)
        expected = rb.array([[-1,  2,  3,  4,  0,  0,  5],
                             [ 0, -1,  0,  2,  0,  0,  3],
                             [ 0,  0,  1,  0,  0,  0,  0],
                             [ 0,  0,  0, -1,  0,  0,  0],
                             [ 2,  4,  5,  8,  1,  0,  9],
                             [ 2,  4,  5,  8,  0, -1,  9],
                             [ 0,  0,  0,  0,  0,  0,  1]])
        assert_equal(A_lambda_2, expected)

    def test_my_use_case_vertex(self):
        #       D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
        A1 = rb.array([[-1, 2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
                       [0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0],  # E
                       [0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],  # R
                       [0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
                       [0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
                       [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0],  # Mi
                       [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # A
                       [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # I
                       [0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0],  # Do
                       [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # Ev
                       [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],  # G
                       [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # Ma
                       [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],  # S
                       [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
                      ], dtype=np.int32)
        #               D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
        u = rb.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3]], dtype=np.int32)
        v = rb.array([[0],  # D
                      [0],  # E
                      [2],  # R
                      [0],  # M
                      [0],  # H
                      [0],  # Mi
                      [0],  # A
                      [0],  # I
                      [2],  # Do
                      [0],  # Ev
                      [0],  # G
                      [0],  # Ma
                      [0],  # S
                      [0]   # Em
                      ],
                     dtype=np.int32)
        A_lambda = A1.vertex_relational_composition(u, v, -1, compute_closure=True)
        assert A_lambda[0][12] == 12
        assert A_lambda[0][13] == 13
        assert A_lambda[0][14] == 6
        assert A_lambda[2][12] == 4
        assert A_lambda[2][13] == 5
        assert A_lambda[2][14] == 2
        assert A_lambda[8][12] == 4
        assert A_lambda[8][13] == 5
        assert A_lambda[8][14] == 2

    def test_cardinality(self):
        A = rb.array([[-1, 2, 3, 4, 0, 0, 5],
                      [0, -1, 0, 2, 0, 0, 3],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [2, 4, 5, 8, 1, 0, 9],
                      [2, 4, 5, 8, 0, -1, 9],
                      [0, 0, 0, 0, 0, 0, 1]])
        cardinality = A.cardinality()
        assert_equal(cardinality['red'], 4)
        assert_equal(cardinality['black'], 3)

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
        results = a.transitive_closure()
        assert_equal(results[0], expected)
        assert_equal(results[1], 3)

    def test_vector_product(self):
        # test rank-1 mutliplication
        u = rb.array([2, 0, 0, 0, 1])
        v = rb.array([3, 0, 1, 0, 0])
        assert_equal(u @ v, 5)
        # test rank-2 multiplication
        u = rb.array([[2, 0, 0, 0, 1]])
        v = rb.array([[3], [0], [1], [0], [0]])
        assert_equal(u @ v, 5)
        # test matrix multiplication
        u = rb.matrix([[2, 0, 0, 0, 1]])
        v = rb.matrix([[3], [0], [1], [0], [0]])
        assert_equal(u @ v, 5)

    def test_vector_matrix_product(self):
        A = rb.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  8,  1]])
        # test rank-1 mutliplication
        u = rb.array([0, 2, 3, 0, 0])
        result = u @ A
        expected = rb.array([0, 2, 3, 4, 0])
        assert_equal(result, expected)

        # test rank-2 multiplication
        u = rb.array([[0, 2, 3, 0, 0]])
        result = u @ A
        assert_equal(result[0], expected)

        A = rb.matrix([[-1,  2,  3,  4,  0],
                       [ 0, -1,  0,  2,  0],
                       [ 0,  0,  1,  0,  0],
                       [ 0,  0,  0, -1,  0],
                       [ 2,  4,  5,  8,  1]])
        u = rb.array([0, 2, 3, 0, 0])
        result = u @ A
        assert_equal(result, expected)

        u = rb.array([[0, 2, 3, 0, 0]])
        result = u @ A
        assert_equal(result[0], expected)

        A = rb.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  8,  1]])
        v = rb.array([0, 0, 9, 0, 0])
        # test rank-1 mutliplication
        result = A @ v
        expected = rb.array([25, 0, 9, 0, 41])
        assert_equal(result, expected)

        # test rank-2 multiplication
        v = rb.array([[0], [0], [9], [0], [0]])
        result = A @ v
        assert_equal(result, expected.reshape((5, 1)))

    def test_vector_matrix_rproduct(self):
        A = rb.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  8,  1]])
        # test rank-1 mutliplication
        u = np.array([0, 2, 3, 0, 0])
        result = u @ A
        expected = rb.array([0, 2, 3, 4, 0])
        assert_equal(result, expected)

        # test rank-2 multiplication
        u = np.array([[0, 2, 3, 0, 0]])
        result = u @ A
        assert_equal(result[0], expected)

        A = rb.matrix([[-1,  2,  3,  4,  0],
                       [ 0, -1,  0,  2,  0],
                       [ 0,  0,  1,  0,  0],
                       [ 0,  0,  0, -1,  0],
                       [ 2,  4,  5,  8,  1]])
        u = np.array([0, 2, 3, 0, 0])
        result = u @ A
        assert_equal(result, expected.reshape((1,5)))

        u = np.array([[0, 2, 3, 0, 0]])
        result = u @ A
        assert_equal(result[0], expected.reshape((1,5)))

        A = np.array([[-1,  2,  3,  4,  0],
                      [ 0, -1,  0,  2,  0],
                      [ 0,  0,  1,  0,  0],
                      [ 0,  0,  0, -1,  0],
                      [ 2,  4,  5,  8,  1]])
        v = rb.array([0, 0, 9, 0, 0])
        # test rank-1 mutliplication
        result = A @ v
        expected = rb.array([25, 0, 9, 0, 41])
        assert_equal(result, expected)

        # test rank-2 multiplication
        v = rb.array([[0], [0], [9], [0], [0]])
        result = A @ v
        assert_equal(result, expected.reshape((5, 1)))

    def test_edge_relational_composition(self):
        R = rb.array([[-1, 0, 3, 0, 0],
                      [ 0,-1, 0, 2, 0],
                      [ 0, 0, 1, 0, 0],
                      [ 0, 0, 0,-1, 0],
                      [ 2, 0, 5, 0, 1]])
        R_lambda = R.edge_relational_composition(0, 1, 2)
        expected = rb.array([[-1, 2, 3, 4, 0],
                             [ 0,-1, 0, 2, 0],
                             [ 0, 0, 1, 0, 0],
                             [ 0, 0, 0,-1, 0],
                             [ 2, 4, 5, 8, 1]])
        assert_equal(R_lambda, expected)

    def test_my_use_case_edge(self):
        #                D  E  R  M  H  Mi A  I Do Ev  G Ma  S Em  J
        R1 = rb.array([[-1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # D
                       [ 0,-1, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],  # E
                       [ 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],  # R
                       [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # M
                       [ 0, 2, 0, 3,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
                       [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 2, 3, 0, 0, 0],  # Mi
                       [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # A
                       [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # I
                       [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 3, 0, 0, 0, 0, 2],  # Do
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Ev
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],  # G
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Ma
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],  # S
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Em
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3,-1]   # J
                       ])
        R = R1.transitive_closure()[0]
        # Missing edge is R -> J, 2
        A_lambda = R.edge_relational_composition(2, 14, 2)
        assert A_lambda[0][12] == 12
        assert A_lambda[0][13] == 13
        assert A_lambda[0][14] == 6
        assert A_lambda[2][12] == 4
        assert A_lambda[2][13] == 5
        assert A_lambda[2][14] == 2
        assert A_lambda[8][12] == 4
        assert A_lambda[8][13] == 5
        assert A_lambda[8][14] == 2



if __name__ == "__main__":
    run_module_suite()