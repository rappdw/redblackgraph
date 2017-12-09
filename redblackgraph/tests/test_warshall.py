import redblackgraph as rb
from numpy.testing import assert_equal


def test_simple():
    a = rb.array([[-1, 2, 3, 0, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 0, 0, 0, 1]])
    expected = rb.array([[-1, 2, 3, 4, 0], [0, -1, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [2, 4, 5, 8, 1]])
    results = rb.warshall(a)
    assert_equal(results, expected)


