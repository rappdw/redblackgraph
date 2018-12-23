from redblackgraph import avos_product, avos_sum

def test_simple_avos_product():
    result = avos_product(7, 4)
    assert result == 28

def test_simple_avos_sum():
    mini = avos_sum(0, 4)
    assert mini == 4

def test_avos_sum():
    operands = []
    sums = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            sums.append(avos_sum(i, j))
            operands.append((i, j))
    errors = ""
    expected = [-1, -1, -1, -1, 0, 1, -1, 1, 1]
    for i in range(len(sums)):
        if sums[i] != expected[i]:
            errors += f"{operands[i][0]} + {operands[i][1]} = {sums[i]}. Expected: {expected[i]}\n"
    assert not errors

def test_product():
    operands = []
    products = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            products.append(avos_product(i, j))
            operands.append((i, j))
    errors = ""
    expected = [-1, 0, -1, 0, 0, 0, -1, 0, 1]
    for i in range(len(products)):
        if products[i] != expected[i]:
            errors += f"{operands[i][0]} * {operands[i][1]} = {products[i]}. Expected: {expected[i]}\n"
    assert not errors

def test_edge_cases_product():
    assert avos_product(9223372036854775807, 2) > 0
    assert avos_product(-1, 1) < 0
    assert avos_product(1, -1) < 0
