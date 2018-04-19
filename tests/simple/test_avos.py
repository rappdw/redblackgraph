from redblackgraph.simple.avos import avos_product, avos_sum

def test_neg():
    result = avos_product(-7, 4)
    assert result == -28
    result = avos_product(7, 4)
    assert result == 28