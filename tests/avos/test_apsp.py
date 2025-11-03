import numpy as np
import pytest
from scipy.sparse import coo_matrix
from numpy.testing import assert_equal

import redblackgraph as rb
from redblackgraph import sparse, RED_ONE, BLACK_ONE
from redblackgraph import reference as ref, core
from redblackgraph.util.test_support import rb_matrix

def core_transitive_closure(A):
    return core.transitive_closure(rb.array(A))

@pytest.mark.parametrize(
    "transitive_closure,result_type",
    [
        (ref.transitive_closure, np.ndarray),
        (core_transitive_closure, rb.array),
        (sparse.transitive_closure_floyd_warshall, rb.array),
        (sparse.transitive_closure_dijkstra, rb.array),
    ]
)
def test_apsp(transitive_closure, result_type):
    # test transitive closure on the example matrix from our notebook
    A = [[-1, 2, 3, 0, 0],
           [ 0,-1, 0, 2, 0],
           [ 0, 0, 1, 0, 0],
           [ 0, 0, 0,-1, 0],
           [ 2, 0, 0, 0, 1]]
    expected = [[-1, 2, 3, 4, 0],
                [ 0,-1, 0, 2, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0,-1, 0],
                [ 2, 4, 5, 8, 1]]
    results = transitive_closure(A)
    assert_equal(results.W, expected)
    assert_equal(results.diameter, 3)
    assert type(results.W) == result_type

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
def test_warshall_core_impl(dtype):
    A = rb.array([[-1, 2, 3, 0, 0],
                  [ 0,-1, 0, 2, 0],
                  [ 0, 0, 1, 0, 0],
                  [ 0, 0, 0,-1, 0],
                  [ 2, 0, 0, 0, 1]], dtype=dtype)
    expected = rb.array([[-1, 2, 3, 4, 0],
                         [ 0,-1, 0, 2, 0],
                         [ 0, 0, 1, 0, 0],
                         [ 0, 0, 0,-1, 0],
                         [ 2, 4, 5, 8, 1]], dtype=dtype)
    results = A.transitive_closure()
    assert_equal(results.W, expected)
    assert_equal(results.diameter, 3)

@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.uint32,
    ]
)
@pytest.mark.parametrize(
    "transitive_closure",
    [
        (sparse.transitive_closure_floyd_warshall),
        (sparse.transitive_closure_dijkstra),
    ])
def test_apsp_sparse_impl(dtype, transitive_closure):
    # NumPy 2.x: use astype() for unsigned dtypes to allow overflow wrapping
    # Diagonal: RED_ONE for vertices 0,1,3,5; BLACK_ONE for 2,4,6
    a_data = np.array([RED_ONE, RED_ONE, BLACK_ONE, RED_ONE, BLACK_ONE, RED_ONE, BLACK_ONE, 2, 3, 2, 3, 2, 3]).astype(dtype)
    a = rb_matrix(
        coo_matrix(
            (
                a_data,
                (
                    [0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2],
                    [0, 1, 2, 3, 4, 5, 6, 1, 2, 5, 6, 3, 4]
                )
            )
        )
    )

    expected = rb.array([[RED_ONE, 2, 3, 6, 7, 4, 5],
                         [ 0, RED_ONE, 0, 0, 0, 2, 3],
                         [ 0, 0, BLACK_ONE, 2, 3, 0, 0],
                         [ 0, 0, 0, RED_ONE, 0, 0, 0],
                         [ 0, 0, 0, 0, BLACK_ONE, 0, 0],
                         [ 0, 0, 0, 0, 0, RED_ONE, 0],
                         [ 0, 0, 0, 0, 0, 0, BLACK_ONE]], dtype=np.int32)

    results = transitive_closure(a)
    # print()
    # print(results)
    assert_equal(results.W, expected)
    assert results.diameter == 2

