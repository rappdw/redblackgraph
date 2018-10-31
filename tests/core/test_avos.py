from redblackgraph import avos_product, avos_sum

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

def test_product():
    products = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            products.append(avos_product(i, j))
    assert products == [-1, 0, -1, 0, 0, 0, -1, 0, 1]