import pytest

import redblackgraph.reference as ref
import redblackgraph.core as core
import redblackgraph.sparse as sparse
import redblackgraph.sparse.csgraph._rbg_math as csgraph

@pytest.mark.parametrize(
    "product",
    [
        (ref.avos_product),
        (core.avos_product),
        (sparse.avos_product),
        (csgraph.py_avos_product)
    ]
)
def test_avos_product(product):
    expected_results = [
#       -1  0   1   2   3   4   5   6   7    8    9   10   11   12   13   14   15
        -1, 0, -1,  2,  3,  4,  5,  6,  7,   8,   9,  10,  11,  12,  13,  14,  15,  # -1
         0, 0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,  #  0
        -1, 0,  1,  2,  3,  4,  5,  6,  7,   8,   9,  10,  11,  12,  13,  14,  15,  #  1
         2, 0,  2,  4,  5,  8,  9, 10, 11,  16,  17,  18,  19,  20,  21,  22,  23,  #  2
         3, 0,  3,  6,  7, 12, 13, 14, 15,  24,  25,  26,  27,  28,  29,  30,  31,  #  3
         4, 0,  4,  8,  9, 16, 17, 18, 19,  32,  33,  34,  35,  36,  37,  38,  39,  #  4
         5, 0,  5, 10, 11, 20, 21, 22, 23,  40,  41,  42,  43,  44,  45,  46,  47,  #  5
         6, 0,  6, 12, 13, 24, 25, 26, 27,  48,  49,  50,  51,  52,  53,  54,  55,  #  6
         7, 0,  7, 14, 15, 28, 29, 30, 31,  56,  57,  58,  59,  60,  61,  62,  63,  #  7
         8, 0,  8, 16, 17, 32, 33, 34, 35,  64,  65,  66,  67,  68,  69,  70,  71,  #  8
         9, 0,  9, 18, 19, 36, 37, 38, 39,  72,  73,  74,  75,  76,  77,  78,  79,  #  9
        10, 0, 10, 20, 21, 40, 41, 42, 43,  80,  81,  82,  83,  84,  85,  86,  87,  # 10
        11, 0, 11, 22, 23, 44, 45, 46, 47,  88,  89,  90,  91,  92,  93,  94,  95,  # 11
        12, 0, 12, 24, 25, 48, 49, 50, 51,  96,  97,  98,  99, 100, 101, 102, 103,  # 12
        13, 0, 13, 26, 27, 52, 53, 54, 55, 104, 105, 106, 107, 108, 109, 110, 111,  # 13
        14, 0, 14, 28, 29, 56, 57, 58, 59, 112, 113, 114, 115, 116, 117, 118, 119,  # 14
        15, 0, 15, 30, 31, 60, 61, 62, 63, 120, 121, 122, 123, 124, 125, 126, 127,  # 15
    ]
    r = range(-1, 16)
    idx = 0
    for x in r:
        for y in r:
            result = product(x, y)
            assert result == expected_results[idx], f"expecting {expected_results[idx]} = {x} * {y}, idx={idx}"
            idx += 1

@pytest.mark.parametrize(
    "sum",
    [
        (ref.avos_sum),
        (core.avos_sum),
        (sparse.avos_sum),
        (csgraph.py_avos_sum)
    ]
)
def test_avos_sum(sum):
    expected_results = [
#       -1   0   1   2   3
        -1, -1, -1, -1, -1, # -1
        -1,  0,  1,  2,  3, #  0
        -1,  1,  1,  1,  1, #  1
        -1,  2,  1,  2,  2, #  2
        -1,  3,  1,  2,  3, #  3
    ]
    r = range(-1, 4)
    idx = 0
    for x in r:
        for y in r:
            result = sum(x, y)
            assert result == expected_results[idx], f"expecting {expected_results[idx]} = {x} + {y}, idx={idx}"
            idx += 1


@pytest.mark.parametrize(
    "product",
    [
        (ref.avos_product),
        (core.avos_product),
        (sparse.avos_product),
#        (csgraph.py_avos_product), # TODO: figure out fuesed type implementation
    ]
)
def test_edge_cases_product(product):
    assert product(9223372036854775807, 2) == 18446744073709551614 # largest ahnen number representable in 64 bits

@pytest.mark.parametrize(
    "product",
    [
        (core.avos_product),
        (sparse.avos_product),
    ]
)
def test_avos_overflow(product):
    x = y = 4294967296
    try:
        _ = product(x, y)
        assert False
    except OverflowError as e:
        pass # this is expected, overflows from 64 bit representation

    x = 9223372036854775807
    y = 3
    try:
        _ = product(x, y)
        assert False
    except OverflowError as e:
        pass # this is expected, collides with using -1 to represent red one

@pytest.mark.parametrize(
    "sum_test",
    [
        (core.avos.avos_sum_test),
    ]
)
def test_avos_sum_impl(sum_test):
    assert sum_test() == 1
