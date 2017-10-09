from redblackgraph.simple import mat_avos


def test_matrix_avos():
    A = [[-1, 2, 3, 0, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 0, 0, 0, 1]]
    expected_result =  [[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 0, 1]]
    result = mat_avos(A, A)
    assert result == expected_result
    expected_result =  [[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 8, 1]]
    result = mat_avos(A, result)
    assert result == expected_result
