from redblackgraph.reference import mat_avos
from redblackgraph import RED_ONE, BLACK_ONE


def test_matrix_avos():
    A = [[RED_ONE,  2,  3,  0,  0],
         [ 0, RED_ONE,  0,  2,  0],
         [ 0,  0,  BLACK_ONE,  0,  0],
         [ 0,  0,  0, RED_ONE,  0],
         [ 2,  0,  0,  0,  BLACK_ONE]]
    expected_result =  [[RED_ONE,  2,  3,  4,  0],
                        [ 0, RED_ONE,  0,  2,  0],
                        [ 0,  0,  BLACK_ONE,  0,  0],
                        [ 0,  0,  0, RED_ONE,  0],
                        [ 2,  4,  5,  0,  BLACK_ONE]]
    result = mat_avos(A, A)

    assert result == expected_result
    expected_result =  [[RED_ONE,  2,  3,  4,  0],
                        [ 0, RED_ONE,  0,  2,  0],
                        [ 0,  0,  BLACK_ONE,  0,  0],
                        [ 0,  0,  0, RED_ONE,  0],
                        [ 2,  4,  5,  8,  BLACK_ONE]]
    result = mat_avos(A, result)
    assert result == expected_result


def test_vec_mat_avos():
    A =  [[RED_ONE,  2,  3,  4,  0],
          [ 0, RED_ONE,  0,  2,  0],
          [ 0,  0,  BLACK_ONE,  0,  0],
          [ 0,  0,  0, RED_ONE,  0],
          [ 2,  4,  5,  0,  BLACK_ONE]]
    u = [[2, 0, 0, 0, 0]]
    v = [[0],
         [3],
         [0],
         [0],
         [0]]

    result = mat_avos(u, A)
    u_lambda = [2, 4, 5, 8, 0]
    assert result[0] == u_lambda
    result = mat_avos(A, v)
    v_lambda = [[5],
                [3],
                [0],
                [0],
                [9]]
    assert result == v_lambda

def test_identity():
    # With parity constraints, identity matrix must respect vertex parity
    A = [[RED_ONE, 2, 3, 0, 0],
         [ 0, RED_ONE, 0, 2, 0],
         [ 0, 0, BLACK_ONE, 0, 0],
         [ 0, 0, 0, RED_ONE, 0],
         [ 2, 0, 0, 0, BLACK_ONE]]
    # Identity matrix matching A's diagonal parity
    I = [[RED_ONE, 0, 0, 0, 0],        # Row 0: RED (even)
         [ 0, RED_ONE, 0, 0, 0],        # Row 1: RED (even)
         [ 0, 0, BLACK_ONE, 0, 0],      # Row 2: BLACK (odd)
         [ 0, 0, 0, RED_ONE, 0],        # Row 3: RED (even)
         [ 0, 0, 0, 0, BLACK_ONE]]      # Row 4: BLACK (odd)

    res = mat_avos(I, A)
    assert A == res
