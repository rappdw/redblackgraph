from redblackgraph.reference import warshall
from numpy.testing import assert_equal


def test_warshall():
    # test transitive closure on the example matrix from our notebook
    Arb = [[-1, 2, 3, 0, 0],
           [ 0,-1, 0, 2, 0],
           [ 0, 0, 1, 0, 0],
           [ 0, 0, 0,-1, 0],
           [ 2, 0, 0, 0, 1]]
    expected_results = [[-1, 2, 3, 4, 0],
                         [ 0,-1, 0, 2, 0],
                         [ 0, 0, 1, 0, 0],
                         [ 0, 0, 0,-1, 0],
                         [ 2, 4, 5, 8, 1]]
    Arb_plus = warshall(Arb)
    assert_equal(Arb_plus.W, expected_results)
    assert 3 == Arb_plus.diameter
