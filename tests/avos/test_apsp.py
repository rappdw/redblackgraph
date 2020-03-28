import numpy as np
import pytest
import redblackgraph as rb
import redblackgraph.reference as ref
import redblackgraph.sparse as sparse
from redblackgraph import rb_matrix
from numpy.testing import assert_equal
from scipy.sparse import coo_matrix


@pytest.mark.parametrize(
    "transitive_closure",
    [
        (ref.transitive_closure),
        (sparse.transitive_closure_floyd_warshall),
        (sparse.transitive_closure_dijkstra),
    ]
)
def test_apsp(transitive_closure):
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
    a = rb_matrix(
        coo_matrix(
            (
                [-1, -1, 1, -1, 1, -1, 1, 2, 3, 2, 3, 2, 3],
                (
                    [0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2],
                    [0, 1, 2, 3, 4, 5, 6, 1, 2, 5, 6, 3, 4]
                )
            ),
        dtype=dtype)
    )

    expected = rb.array([[-1, 2, 3, 6, 7, 4, 5],
                         [ 0,-1, 0, 0, 0, 2, 3],
                         [ 0, 0, 1, 2, 3, 0, 0],
                         [ 0, 0, 0,-1, 0, 0, 0],
                         [ 0, 0, 0, 0, 1, 0, 0],
                         [ 0, 0, 0, 0, 0,-1, 0],
                         [ 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32)

    results = transitive_closure(a)
    # print()
    # print(results)
    assert_equal(results.W, expected)
    assert results.diameter == 2

