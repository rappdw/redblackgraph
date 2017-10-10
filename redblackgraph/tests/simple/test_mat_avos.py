from redblackgraph.simple import mat_avos


def test_matrix_avos():
    A = [[-1, 2, 3, 0, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 0, 0, 0, 1]]
    expected_result =  [[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 0, 1]]
    result = mat_avos(A, A)
    assert result == expected_result
    expected_result =  [[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 8, 1]]
    result = mat_avos(A, result)
    assert result == expected_result


def test_vec_mat_avos():
    A =  [[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 0, 1]]
    u = [[2, 0, 0, 0, 0]]
    v = [[0], [0], [0], [0], [0]]

    result = mat_avos(u, A)
    assert result[0] == [2, 4, 5, 8, 0]
    result = mat_avos(A, v)
    assert not [e for e in result if not e[0] == 0]
