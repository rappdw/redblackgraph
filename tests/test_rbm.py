import pytest
from scipy.sparse import coo_matrix
from redblackgraph import rb_matrix, Color


MATRICES = [
    # elementary case
    # {
    #     # 0 - me
    #     # 1 - father
    #     # 2 - mother
    #     # 3 - paternal grandfather
    #     'input': rb_matrix((
    #         [0, 2, 3, 0, 2, 1, 0],
    #         [0, 1, 2, 1, 3, 2, 3],
    #         [0, 3, 5, 6, 7]
    #     )),
    #     'expected': rb_matrix((
    #         [0, 2, 3, 4, 0, 2, 1, 0],
    #         [0, 1, 2, 3, 1, 3, 2, 3],
    #         [0, 4, 6, 7, 8]
    #     )),
    #     'iterations': 1
    # },
    {
        # 0 - me
        # 1 - father
        # 2 - mother
        # 3 - paternal grandfather
        'input': rb_matrix(
            coo_matrix(
                (
                        [-1, 2, 3, -1, 2, 1, -1],
                    (
                        [ 0, 0, 0,  1, 1, 2,  3],
                        [ 0, 1, 2,  1, 3, 2,  3]
                    )
                ),
                shape=(4, 4)
            )
        ),
        'expected': rb_matrix(
            coo_matrix(
                (
                        [-1, 2, 3, -1, 2, 1, -1, 4],
                    (
                        [ 0, 0, 0,  1, 1, 2,  3, 0],
                        [ 0, 1, 2,  1, 3, 2,  3, 3]
                    )
                ),
                shape=(4,4)
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

        'input': rb_matrix((
            [0, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 0, 1],
            [0, 1, 2, 1, 5, 6, 2, 3, 4, 3, 4, 5, 6],
            [0, 3, 6, 9, 10, 11, 12, 13]
        )),
        'expected': rb_matrix((
            [7, 6, 5, 4, 3, 2, 3, 2, 3, 2, 1, 1, 1],
            [4, 3, 6, 5, 2, 1, 6, 5, 4, 3, 2, 4, 6],
            [0, 6, 8, 11, 11, 12, 12, 13]
        )),
        'iterations': 1
    },
    # moderately complex case
    {
        'input': rb_matrix((
            [0, 3, 1, 2, 3, 1, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 1, 3, 1, 1, 2, 3, 0, 0, 1, 1, 0, 2, 3, 0,
             3, 2, 1, 1],
            [0, 7, 1, 5, 12, 2, 10, 16, 3, 13, 14, 4, 5, 6, 10, 16, 7, 17, 8, 9, 3, 9, 10, 11, 12, 13,
             14, 4, 8, 15, 1, 15,
             16, 17],
            [0, 2, 5, 8, 11, 12, 13, 16, 18, 19, 20, 23, 24, 25, 26, 27, 30, 33, 34]
        )),
        'expected': rb_matrix((
            [0, 3, 7, 1, 2, 3, 7, 1, 4, 5, 2, 6, 3, 0, 3, 2, 0, 0, 7, 4, 0, 5, 2, 6, 3, 1, 3, 1, 1,
             2, 3, 0, 5, 4, 0, 1, 1,
             0, 2, 3, 0, 3, 4, 6, 5, 7, 2, 1, 1],
            [0, 7, 17, 1, 5, 12, 1, 2, 3, 9, 10, 15, 16, 3, 13, 14, 4, 5, 1, 3, 6, 9, 10, 15, 16, 7,
             17, 8, 9, 3, 9, 10, 13,
             14, 11, 12, 13, 14, 4, 8, 15, 1, 4, 5, 8, 12, 15, 16, 17],
            [0, 3, 6, 13, 16, 17, 18, 25, 27, 28, 29, 34, 35, 36, 37, 38, 41, 48, 49]
        )),
        'iterations': 1
    },
    # moderately complex case (second iteration
    {
        'input': rb_matrix((
            [0, 3, 7, 1, 2, 3, 7, 1, 4, 5, 2, 6, 3, 0, 3, 2, 0, 0, 7, 4, 0, 5, 2, 6, 3, 1, 3, 1, 1, 2,
             3, 0, 5, 4, 0, 1, 1,
             0, 2, 3, 0, 3, 4, 6, 5, 7, 2, 1, 1],
            [0, 7, 17, 1, 5, 12, 1, 2, 3, 9, 10, 15, 16, 3, 13, 14, 4, 5, 1, 3, 6, 9, 10, 15, 16, 7,
             17, 8, 9, 3, 9, 10, 13,
             14, 11, 12, 13, 14, 4, 8, 15, 1, 4, 5, 8, 12, 15, 16, 17],
            [0, 3, 6, 13, 16, 17, 18, 25, 27, 28, 29, 34, 35, 36, 37, 38, 41, 48, 49]
        )),
        'expected': rb_matrix((
            [0, 3, 7, 1, 2, 3, 7, 1, 4, 12, 14, 13, 5, 2, 15, 9, 8, 6, 3, 0, 3, 2, 0, 0, 7, 4, 12,
             14, 0, 13, 5, 2, 15, 9,
             8, 6, 3, 1, 3, 1, 1, 2, 3, 0, 5, 4, 0, 1, 1, 0, 2, 3, 0, 3, 4, 6, 5, 7, 2, 1, 1],
            [0, 7, 17, 1, 5, 12, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 3, 13, 14, 4, 5, 1, 3,
             4, 5, 6, 8, 9, 10, 12,
             13, 14, 15, 16, 7, 17, 8, 9, 3, 9, 10, 13, 14, 11, 12, 13, 14, 4, 8, 15, 1, 4, 5, 8,
             12, 15, 16, 17],
            [0, 3, 6, 19, 22, 23, 24, 37, 39, 40, 41, 46, 47, 48, 49, 50, 53, 60, 61]
        )),
        'iterations': 1
    },
    {
        'input': rb_matrix((
            [0, 3, 1, 2, 3, 1, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 1, 3, 1, 1, 2, 3, 0, 0, 1, 1, 0, 2, 3, 0,
             3, 2, 1, 1],
            [0, 7, 1, 5, 12, 2, 10, 16, 3, 13, 14, 4, 5, 6, 10, 16, 7, 17, 8, 9, 3, 9, 10, 11, 12, 13,
             14, 4, 8, 15, 1, 15,
             16, 17],
            [0, 2, 5, 8, 11, 12, 13, 16, 18, 19, 20, 23, 24, 25, 26, 27, 30, 33, 34]
        )),
        'expected': rb_matrix((
            [0, 3, 7, 1, 2, 3, 7, 1, 4, 12, 14, 13, 5, 2, 15, 9, 8, 6, 3, 0, 3, 2, 0, 0, 7, 4, 12,
             14, 0, 13, 5, 2, 15, 9,
             8, 6, 3, 1, 3, 1, 1, 2, 3, 0, 5, 4, 0, 1, 1, 0, 2, 3, 0, 3, 4, 6, 5, 7, 2, 1, 1],
            [0, 7, 17, 1, 5, 12, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 3, 13, 14, 4, 5, 1, 3,
             4, 5, 6, 8, 9, 10, 12,
             13, 14, 15, 16, 7, 17, 8, 9, 3, 9, 10, 13, 14, 11, 12, 13, 14, 4, 8, 15, 1, 4, 5, 8,
             12, 15, 16, 17],
            [0, 3, 6, 19, 22, 23, 24, 37, 39, 40, 41, 46, 47, 48, 49, 50, 53, 60, 61]
        )),
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

