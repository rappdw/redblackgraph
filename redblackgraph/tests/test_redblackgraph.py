import pytest
import redblackgraph
from scipy.sparse import csr_matrix

GENERATION_TUPLES = [
    # (pedigree number, expected gneeration)
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 1),
    (4, 2),
    (7, 2),
    (8, 3),
    (15, 3),
    (16, 4)
]

AVOS_TUPLES = [
    #  (person1, person2, expected relationship)
    # validate self (male)
    (0, 0, 0),
    # validate self (female)
    (1, 1, 1),
    # validate self (male) parents
    (0, 2, 2),
    (0, 3, 3),
    # validate self (female) parents
    (1, 2, 2),
    (1, 3, 3),
    # validate father's parents
    (2, 2, 4),
    (2, 3, 5),
    # validate mother's parents
    (3, 2, 6),
    (3, 3, 7),
    # validate father's grandparents
    (2, 4, 8),
    (2, 5, 9),
    (2, 6, 10),
    (2, 7, 11),
    (2, 8, 16),
    (2, 9, 17),
    (2, 10, 18),
    (2, 11, 19),
    (2, 12, 20),
    (2, 13, 21),
    (2, 14, 22),
    (2, 15, 23),
    # validate mother's grandparents
    (3, 4, 12),
    (3, 5, 13),
    (3, 6, 14),
    (3, 7, 15),
    (3, 8, 24),
    (3, 9, 25),
    (3, 10, 26),
    (3, 11, 27),
    (3, 12, 28),
    (3, 13, 29),
    (3, 14, 30),
    (3, 15, 31),
    # validate paternal grandfather's parents
    (4, 2, 8),
    (4, 3, 9),
    # validate paternal grandmother's parents
    (5, 2, 10),
    (5, 3, 11),
    # validate maternal grandfather's parents
    (6, 2, 12),
    (6, 3, 13),
    # validate maternal grandmother's parents
    (7, 2, 14),
    (7, 3, 15)
]

AVOS_INVALID_TUPLES = [
    # (person1, person2)
    (0, 1),
    (1, 0)
]

MATRICES = [
    # elementary case
    {
        # 0 - me
        # 1 - father
        # 2 - mother
        # 3 - paternal grandfather
        'input': csr_matrix((
            [0, 2, 3, 0, 2, 1, 0],
            [0, 1, 2, 1, 3, 2, 3],
            [0, 3, 5, 6, 7]
        )),
        'expected': csr_matrix((
            [0, 2, 3, 4, 0, 2, 1, 0],
            [0, 1, 2, 3, 1, 3, 2, 3],
            [0, 4, 6, 7, 8]
        )),
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

        'input': csr_matrix((
            [0, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 0, 1],
            [0, 1, 2, 1, 5, 6, 2, 3, 4, 3, 4, 5, 6],
            [0, 3, 6, 9, 10, 11, 12, 13]
        )),
        'expected': csr_matrix((
            [7, 6, 5, 4, 3, 2, 3, 2, 3, 2, 1, 1, 1],
            [4, 3, 6, 5, 2, 1, 6, 5, 4, 3, 2, 4, 6],
            [0, 6, 8, 11, 11, 12, 12, 13]
        )),
        'iterations': 1
    },
    # moderately complex case
    {
        'input': csr_matrix((
            [0, 3, 1, 2, 3, 1, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 1, 3, 1, 1, 2, 3, 0, 0, 1, 1, 0, 2, 3, 0,
             3, 2, 1, 1],
            [0, 7, 1, 5, 12, 2, 10, 16, 3, 13, 14, 4, 5, 6, 10, 16, 7, 17, 8, 9, 3, 9, 10, 11, 12, 13,
             14, 4, 8, 15, 1, 15,
             16, 17],
            [0, 2, 5, 8, 11, 12, 13, 16, 18, 19, 20, 23, 24, 25, 26, 27, 30, 33, 34]
        )),
        'expected': csr_matrix((
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
        'input': csr_matrix((
            [0, 3, 7, 1, 2, 3, 7, 1, 4, 5, 2, 6, 3, 0, 3, 2, 0, 0, 7, 4, 0, 5, 2, 6, 3, 1, 3, 1, 1, 2,
             3, 0, 5, 4, 0, 1, 1,
             0, 2, 3, 0, 3, 4, 6, 5, 7, 2, 1, 1],
            [0, 7, 17, 1, 5, 12, 1, 2, 3, 9, 10, 15, 16, 3, 13, 14, 4, 5, 1, 3, 6, 9, 10, 15, 16, 7,
             17, 8, 9, 3, 9, 10, 13,
             14, 11, 12, 13, 14, 4, 8, 15, 1, 4, 5, 8, 12, 15, 16, 17],
            [0, 3, 6, 13, 16, 17, 18, 25, 27, 28, 29, 34, 35, 36, 37, 38, 41, 48, 49]
        )),
        'expected': csr_matrix((
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
        'input': csr_matrix((
            [0, 3, 1, 2, 3, 1, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 1, 3, 1, 1, 2, 3, 0, 0, 1, 1, 0, 2, 3, 0,
             3, 2, 1, 1],
            [0, 7, 1, 5, 12, 2, 10, 16, 3, 13, 14, 4, 5, 6, 10, 16, 7, 17, 8, 9, 3, 9, 10, 11, 12, 13,
             14, 4, 8, 15, 1, 15,
             16, 17],
            [0, 2, 5, 8, 11, 12, 13, 16, 18, 19, 20, 23, 24, 25, 26, 27, 30, 33, 34]
        )),
        'expected': csr_matrix((
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


@pytest.mark.parametrize('generation_tuple', GENERATION_TUPLES)
def test_generation(generation_tuple):
    assert redblackgraph.generation(generation_tuple[0]) == generation_tuple[1]


def test_invalid_generation():
    try:
        redblackgraph.generation(-1)
    except ValueError:
        pass  # this is what we expect
    else:
        assert False


@pytest.mark.parametrize('avos_tuple', AVOS_TUPLES)
def test_avos(avos_tuple):
    assert redblackgraph.avos(avos_tuple[0], avos_tuple[1]) == avos_tuple[2]


@pytest.mark.parametrize('avos_tuple', AVOS_INVALID_TUPLES)
def test_invalid_avos(avos_tuple):
    try:
        redblackgraph.avos(avos_tuple[0], avos_tuple[1])
    except ValueError:
        pass  # this is what we expect
    else:
        assert False


def matrx_equality_test(m1, m2):
    test_ne = m1 != m2
    if test_ne.nnz != 0:
        print_matrix("Falied-m1", m1)
        print_matrix("Falied-m2", m2)
        print(test_ne)
    assert test_ne.nnz == 0

def print_matrix(label, m):
    sorted = m.tocsc().tocsr()
    # there is a bug on the sparse matrix printing that is exposed by this
    # test case... to work around it set maxprint to a large number
    sorted.maxprint = 1000
    print('{}: \n{}'.format(label, sorted))


@pytest.mark.parametrize('matrix', MATRICES)
def test_expand(matrix):
    A = matrix['input']
    for i in range(0, matrix['iterations']):
        A = redblackgraph.expand(A)
    expected_results = matrix['expected']
    matrx_equality_test(A, expected_results)
