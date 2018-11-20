from redblackgraph.reference.topological_sort import topological_sort

def test_simple():
    A = [[-1, 2, 3, 0, 0],
         [ 0,-1, 0, 2, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0,-1, 0],
         [ 2, 0, 0, 0, 1]]

    order = topological_sort(A)
    assert order[0] == 4
    assert order[1] == 0
    assert order[-1] == 2 or order[-1] == 3