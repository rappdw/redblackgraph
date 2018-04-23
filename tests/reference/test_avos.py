from redblackgraph.reference.avos import avos_product, avos_sum, compute_sign


def test_sign():
    pairs = [
        (-2, -4),
        (-2, -1),
        (-1, -2),
        (-1, -1),
        (-2, 4),
        (-1, 4),
        (4, -2),
        (4, -1),
        (2, 4),
        (-1, 1),
        (1, -1),
    ]
    expected = [
        None,
        -1,
        -1,
        -1,
        None,
        1,
        None,
        1,
        1,
        -1,
        -1
    ]
    for expected, pair in zip(expected, pairs):
        assert expected == compute_sign(*pair)

def test_simple_avos_product():
    result = avos_product(7, 4)
    assert result == 28

def test_simple_avos_sum():
    mini = avos_sum(0, 4)
    assert mini == 4