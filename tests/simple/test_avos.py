from redblackgraph.simple.avos import avos_product, avos_sum

def test_simple_avos_product():
    result = avos_product(-7, 4)
    assert result == -28
    result = avos_product(7, 4)
    assert result == 28

def test_simple_avos_sum():
    mini = avos_sum(0, 4)
    assert mini == 4