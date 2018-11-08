import pytest
from scipy.sparse import coo_matrix
from redblackgraph import rb_matrix, Color


MATRICES = [
    # elementary case
    {
        # 0 - me
        # 1 - father
        # 2 - mother
        # 3 - paternal grandfather
        'input': rb_matrix(
            coo_matrix(
                (
                        [-1, 2, 3,-1, 2, 1,-1],
                    (
                        [ 0, 0, 0, 1, 1, 2, 3],
                        [ 0, 1, 2, 1, 3, 2, 3]
                    )
                )
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1, 2, 3,-1, 2, 1,-1, 4],
                    (
                        [ 0, 0, 0, 1, 1, 2, 3, 0],
                        [ 0, 1, 2, 1, 3, 2, 3, 3]
                    )
                )
            )
        ),
        'iterations': 1
    },
    # simple case
    {
        # 0 - me
        # 1 - father
        # 2 - mother
        # 3 - maternal grandfather
        # 4 - maternal grandmother
        # 5 - paternal grandfather
        # 6 - paternal grandmother

        'input': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 2, 3],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2],
                        [ 0, 1, 2, 3, 4, 5, 6, 1, 2, 5, 6, 3, 4]
                    )
                )
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 2, 3, 6, 7, 4, 5],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0],
                        [ 0, 1, 2, 3, 4, 5, 6, 1, 2, 5, 6, 3, 4, 3, 4, 5, 6]
                    )
                )
            )
        ),
        'iterations': 1
    },
    # My use case
    {
        # from test_redblack.py
        #                      0   1   2   3   4   5   6   7   8   9  10  11  12  13
        #     #                D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
        #     A1 = rb.array([[-1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D  0
        #                    [ 0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0],  # E  1
        #                    [ 0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],  # R  2
        #                    [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M  3
        #                    [ 0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H  4
        #                    [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0],  # Mi 5
        #                    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # A  6
        #                    [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # I  7
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0],  # Do 8
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # Ev 9
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],  # G  10
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # Ma 11
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],  # S  12
        #                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]   # Em 13
        #                   ], dtype=np.int32)

        'input': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9]
                    )
                )
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 4, 5, 7, 4, 5, 4, 5],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8, 0, 0, 0, 1, 1, 4, 4],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9, 5, 6, 7,10,11, 5, 6]
                    )
                )
            )
        ),
        'iterations': 1
    },
    # moderately complex case (second iteration)
    {
        'input': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 4, 5, 7, 4, 5, 4, 5],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8, 0, 0, 0, 1, 1, 4, 4],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9, 5, 6, 7,10,11, 5, 6]
                    )
                )
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 4, 5, 7, 4, 5, 4, 5, 8, 9, 8, 9],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8, 0, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9, 5, 6, 7,10,11, 5, 6,10,11,10,11]
                    )
                )
            )
        ),
        'iterations': 1
    },
    {
        'input': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 4, 5, 7, 4, 5, 4, 5, 8, 9, 8, 9],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8, 0, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9, 5, 6, 7,10,11, 5, 6,10,11,10,11]
                    )
                )
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 2, 3, 2, 3, 3, 2, 3, 2, 3, 3, 4, 5, 7, 4, 5, 4, 5, 8, 9, 8, 9],
                    (
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 0, 0, 1, 1, 2, 4, 4, 5, 5, 8, 0, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4],
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13, 1, 2, 5, 6, 7, 1, 3,10,11, 9, 5, 6, 7,10,11, 5, 6,10,11,10,11]
                    )
                )
            )
        ),
        'iterations': 3 # ensure that an extra squaring doesn't change a valid result
    },
    # Need a test case where a common ancestor can be reached from two paths Amalie Welke grandparents use case
]


def matrx_equality_test(actual, expected):
    test_ne = actual != expected
    if test_ne.nnz != 0:
        print()
        print_matrix("Falied-actual", actual)
        print_matrix("Falied-expected", expected)
        print_matrix("Failed-ne", test_ne)
    assert test_ne.nnz == 0

def print_matrix(label, m):
    sorted = m.tocsc().tocsr()
    # there is a bug on the sparse matrix printing that is exposed by this
    # test case... to work around it set maxprint to a large number
    sorted.maxprint = 1000
    print('{}: \n{}'.format(label, sorted))


@pytest.mark.parametrize('matrix', MATRICES)
def test_rb_matrix_square(matrix):
    A = matrix['input']
    for i in range(0, matrix['iterations']):
        A = A * A
    expected_results = matrix['expected']
    matrx_equality_test(A, expected_results)


def test_relational_composition():
    A = rb_matrix(coo_matrix(([0, 0, 2, 0, 1, 3, 1, 1, 3, 0], ([0, 1, 1, 2, 3, 3, 4, 5, 6, 6], [0, 1, 2, 2, 3, 5, 4, 5, 4, 6])), shape=(7, 7)).tocsr())
    u = rb_matrix(coo_matrix(([2, 3], ([0, 0], [1, 3])), shape=(1, 7)).tocsr())
    v = rb_matrix(coo_matrix(([2, 2], ([0, 4], [0, 0])), shape=(7, 1)).tocsr())

    B = A.rc(u, v, Color.RED)
    # TODO: add assertions to validate the result

def test_transitive_closure():
    # TODO: need to implement warshall algorithm as described in the Jupyter notebook and then test
    pass
