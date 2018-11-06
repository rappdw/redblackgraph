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

def test_avos_sum():
    sums = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            sums.append(avos_sum(i, j))
    assert sums == [-1, -1, 0, -1, 0, 1, 0, 1, 1]

def test_sign2():
    signs = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            signs.append(compute_sign(i, j))
    assert signs == [-1, 1, -1, 1, 1, 1, -1, 1, 1]

def test_product():
    products = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            products.append(avos_product(i, j))
    assert products == [-1, 0, -1, 0, 0, 0, -1, 0, 1]

def test_signed_product():
    for i in range(10):
        for j in range(10):
            try:
                product = avos_product(i, j)
                neg_product = -avos_product(-i, -j)
                assert product == neg_product
            except:
                print(f"error for i: {i}, j: {j}, prod: {product}, -prod: {neg_product}")
    raise ValueError("I believe this test case is failing... investigate")