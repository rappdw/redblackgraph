from redblackgraph.simple import find_components


def test_find_components():
    A = [[-1, 2, 3, 0, 0, 0, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 0, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 2, 0, 0],
         [ 0, 0, 0, 0,-1, 2, 3],
         [ 0, 0, 0, 0, 0,-1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    u = find_components(A)
    assert u == ([1, 1, 1, 2, 2, 2, 2], 2)
