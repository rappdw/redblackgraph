import pytest
from redblackgraph.operators import generation, acc, avos

GENERATION_TUPLES = [
    # (pedigree number, expected generation)
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

@pytest.mark.parametrize('generation_tuple', GENERATION_TUPLES)
def test_generation(generation_tuple):
    assert generation(generation_tuple[0]) == generation_tuple[1]


def test_invalid_generation():
    try:
        generation(-1)
    except ValueError:
        pass  # this is what we expect
    else:
        assert False


@pytest.mark.parametrize('avos_tuple', AVOS_TUPLES)
def test_avos(avos_tuple):
    assert (avos_tuple[0] <<avos>> avos_tuple[1]) == avos_tuple[2]


@pytest.mark.parametrize('avos_tuple', AVOS_INVALID_TUPLES)
def test_invalid_avos(avos_tuple):
    try:
        _ = avos_tuple[0] <<avos>> avos_tuple[1]
    except ValueError:
        pass  # this is what we expect
    else:
        assert False


