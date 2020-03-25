import numpy as np
import redblackgraph.reference as ref
import redblackgraph.sparse.csgraph as sparse
import pytest

from numpy.testing import assert_equal


DTYPE = np.int32  # this should be the same as the definition in parameters.pxi
ITYPE = np.int32

def permute(A, p, assume_upper_triangular=False):
    return sparse.permute(np.array(A, dtype=DTYPE), np.array(p, dtype=ITYPE), assume_upper_triangular)


def execute_case(A, p, expected_false, expected_true):
    result = permute(A, p)
    assert_equal(result, expected_false)
    result = permute(A, p, True)
    assert_equal(result, expected_true)


@pytest.mark.parametrize(
    "permute",
    [
        (ref.permute),
        (permute)
    ]
)
def test_practical_case(permute):
    # first consider a practical use case
    A = [
        [-1, 0, 0, 0, 3, 2, 7, 0, 0, 4],
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 3,-1, 0, 5, 4,11, 0, 0, 8],
        [ 0, 0, 0, 1, 0, 0, 0, 2, 3, 0],
        [ 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 2],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    ]

    expected = [
      [-1, 2, 4, 5, 3, 8,11, 0, 0, 0],
      [ 0,-1, 2, 3, 0, 4, 7, 0, 0, 0],
      [ 0, 0,-1, 0, 0, 2, 0, 0, 0, 0],
      [ 0, 0, 0, 1, 0, 0, 3, 0, 0, 0],
      [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
      [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]

    p = [2, 0, 5, 4, 1, 9, 6, 3, 7, 8]

    P = permute(A, p)
    assert_equal(P, expected)
    P = permute(A, p, assume_upper_triangular=True)
    assert_equal(P, expected)


@pytest.mark.parametrize(
    "permute",
    [
        (ref.permute),
        (permute)
    ]
)
def test_all_mathematical_possibilities(permute):
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    p = [0, 1, 2] # identity
    expected_true = [
        [1, 2, 3],
        [0, 5, 6],
        [0, 0, 9]
    ]
    execute_case(A, p, A, expected_true)

    p = [0, 2, 1] # swap vertex 1 & 2
    expected_false = [
        [1, 3, 2],
        [7, 9, 8],
        [4, 6, 5]
    ]
    expected_true = [
        [1, 3, 2],
        [0, 9, 8],
        [0, 0, 5]
    ]
    execute_case(A, p, expected_false, expected_true)

    p = [0, 2, 1] # swap vertex 1 & 2
    expected_false = [
        [1, 3, 2],
        [7, 9, 8],
        [4, 6, 5]
    ]
    expected_true = [
        [1, 3, 2],
        [0, 9, 8],
        [0, 0, 5]
    ]
    execute_case(A, p, expected_false, expected_true)

    p = [1, 0, 2] # swap vertex 0 & 1
    expected_false = [
        [5, 4, 6],
        [2, 1, 3],
        [8, 7, 9]
    ]
    expected_true = [
        [5, 4, 6],
        [0, 1, 3],
        [0, 0, 9]
    ]
    execute_case(A, p, expected_false, expected_true)

    p = [1, 2, 0] # swap vertex 1 into 0, 2 into 1, 0 into 3
    expected_false = [
        [5, 6, 4],
        [8, 9, 7],
        [2, 3, 1]
    ]
    expected_true = [
        [5, 6, 4],
        [0, 9, 7],
        [0, 0, 1]
    ]
    execute_case(A, p, expected_false, expected_true)

    p = [2, 1, 0] # revers order of vertices
    expected_false = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ]
    expected_true = [
        [9, 8, 7],
        [0, 5, 4],
        [0, 0, 1]
    ]
    execute_case(A, p, expected_false, expected_true)

    p = [2, 0, 1] # swap 2 into 0, 0 into 1 and 1 into 2
    expected_false = [
        [9, 7, 8],
        [3, 1, 2],
        [6, 4, 5]
    ]
    expected_true = [
        [9, 7, 8],
        [0, 1, 2],
        [0, 0, 5]
    ]
    execute_case(A, p, expected_false, expected_true)

