from redblackgraph.reference import mat_avos


def test_matrix_avos():
    A = [[-1,  2,  3,  0,  0],
         [ 0, -1,  0,  2,  0],
         [ 0,  0,  1,  0,  0],
         [ 0,  0,  0, -1,  0],
         [ 2,  0,  0,  0,  1]]
    expected_result =  [[-1,  2,  3,  4,  0],
                        [ 0, -1,  0,  2,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0,  0, -1,  0],
                        [ 2,  4,  5,  0,  1]]
    result = mat_avos(A, A)

    assert result == expected_result
    expected_result =  [[-1,  2,  3,  4,  0],
                        [ 0, -1,  0,  2,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0,  0, -1,  0],
                        [ 2,  4,  5,  8,  1]]
    result = mat_avos(A, result)
    assert result == expected_result


def test_vec_mat_avos():
    A =  [[-1,  2,  3,  4,  0],
          [ 0, -1,  0,  2,  0],
          [ 0,  0,  1,  0,  0],
          [ 0,  0,  0, -1,  0],
          [ 2,  4,  5,  0,  1]]
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
    A = [[-1, 2, 3, 0, 0],
         [ 0,-1, 0, 2, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0, -1, 0],
         [ 2, 0, 0, 0, 1]]
    I = [[ 1, 0, 0, 0, 0],
         [ 0, 1, 0, 0, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 1]]

    res = mat_avos(I, A)
    assert A == res
