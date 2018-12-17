import numpy as np
from redblackgraph import einsum, avos_sum, avos_product
from numpy.testing import (assert_equal)
import pytest

@pytest.mark.parametrize("dtype", [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64
])
def test_avos(dtype):
    # test simple avos matmul
    A = np.array([[-1,2, 3, 0, 0],
                  [0,-1, 0, 2, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0,-1, 0],
                  [2, 0, 0, 0, 1]], dtype=dtype)
    S = np.array([[-1, 2, 3, 4, 0],
                  [0,-1, 0, 2, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0,-1, 0],
                  [2, 4, 5, 0, 1]], dtype=dtype)
    assert_equal(einsum('ij, jk', A, A, avos=True), S)

    # test vector mat mul
    A_star = np.array([[-1,2, 3, 4, 0],
                       [0,-1, 0, 2, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0,-1, 0],
                       [2, 4, 5, 8, 1]], dtype=dtype)
    u = np.array([2, 0, 0, 0, 0], dtype=dtype)
    v = np.array([0, 3, 0, 0, 0], dtype=dtype)
    u_lambda = np.array([2, 4, 5, 8, 0])
    v_lambda = np.array([5, 3, 0, 0, 9])
    assert_equal(einsum('ij,jk', u.reshape(1, 5), A_star, avos=True), u_lambda.reshape(1, 5))
    assert_equal(einsum('ij,jk', A_star, v.reshape(5, 1), avos=True), v_lambda.reshape(5, 1))
    assert_equal(einsum('i,...ij', u, A_star, avos=True), u_lambda)
    assert_equal(einsum('...i,i', A_star, v, avos=True), v_lambda)

@pytest.mark.parametrize("dtype", [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64
])
def test_identity(dtype):
    A = np.array([[-1, 2, 3, 0, 0],
                  [ 0,-1, 0, 2, 0],
                  [ 0, 0, 1, 0, 0],
                  [ 0, 0, 0,-1, 0],
                  [ 2, 0, 0, 0, 1]], dtype=dtype)
    I = np.array([[ 1, 0, 0, 0, 0],
                  [ 0, 1, 0, 0, 0],
                  [ 0, 0, 1, 0, 0],
                  [ 0, 0, 0, 1, 0],
                  [ 0, 0, 0, 0, 1]], dtype=dtype)

    res = einsum('ij,jk', I, A, avos=True)
    assert_equal(A, res)

def test_avos_sum():
    operands = []
    sums = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            sums.append(avos_sum(i, j))
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
    errors = ""
    expected = [-1, 0, -1, 0, 0, 0, -1, 0, 1]
    for i in range(len(products)):
        if products[i] != expected[i]:
            errors += f"{operands[i][0]} * {operands[i][1]} = {products[i]}. Expected: {expected[i]}\n"
    assert not errors
