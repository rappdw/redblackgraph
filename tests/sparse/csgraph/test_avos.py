from redblackgraph.sparse.csgraph._rbg_math import py_avos_product, py_avos_sum

def test_simple_avos_product():
    result = py_avos_product(7, 4)
    assert result == 28

def test_simple_avos_sum():
    mini = py_avos_sum(0, 4)
    assert mini == 4

def test_avos_sum():
    operands = []
    sums = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            sums.append(py_avos_sum(i, j))
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
            products.append(py_avos_product(i, j))
            operands.append((i, j))
    errors = ""
    expected = [-1, 0, -1, 0, 0, 0, -1, 0, 1]
    for i in range(len(products)):
        if products[i] != expected[i]:
            errors += f"{operands[i][0]} * {operands[i][1]} = {products[i]}. Expected: {expected[i]}\n"
    assert not errors

def test_edge_cases_product():
    assert py_avos_product(9223372036854775807, 2) == 18446744073709551614 # largest ahnen number representable in 64 bits
    assert py_avos_product(-1, 1) == -1
    assert py_avos_product(1, -1) == -1

def test_avos_overflow():
    x = y = 4294967296
    try:
        _ = py_avos_product(x, y)
        assert False
    except OverflowError as e:
        pass # this is expected, overflows from 64 bit representation

    x = 9223372036854775807
    y = 3
    try:
        z = py_avos_product(x, y)
        print(z)
        assert False
    except OverflowError as e:
        pass # this is expected, collides with using -1 to represent red node

