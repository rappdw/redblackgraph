import numpy as np
import pytest
from numpy.testing import assert_equal
from redblackgraph import rb_matrix
from redblackgraph.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix

import redblackgraph as rb


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("method", ['FW', 'D']) # TODO: should we even consider Bellman-Ford and Johnson?, 'BF', 'J'])
def test_warshall(dtype, method):
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

    results = shortest_path(a, method=method, directed=True, overwrite=False)
    # print()
    # print(results)
    assert_equal(results, expected)
