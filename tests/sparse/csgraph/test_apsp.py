import numpy as np
import pytest
from numpy.testing import assert_equal
from redblackgraph import rb_matrix
from redblackgraph.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix

import redblackgraph as rb


@pytest.mark.parametrize("dtype", [
    np.int32,
    np.uint32,
])
def test_warshall(dtype):
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
    print()
    print(a)

    results = shortest_path(a, method='FW', directed=True, overwrite=False)
    print()
    print(results)
    # a = rb.array([[-1,  2,  3,  0,  0],
    #               [ 0, -1,  0,  2,  0],
    #               [ 0,  0,  1,  0,  0],
    #               [ 0,  0,  0, -1,  0],
    #               [ 2,  0,  0,  0,  1]], dtype=dtype)
    # expected = rb.array([[-1,  2,  3,  4,  0],
    #                      [ 0, -1,  0,  2,  0],
    #                      [ 0,  0,  1,  0,  0],
    #                      [ 0,  0,  0, -1,  0],
    #                      [ 2,  4,  5,  8,  1]], dtype=dtype)
    # results = a.transitive_closure()
    # assert_equal(results.W, expected)
    # assert_equal(results.diameter, 3)