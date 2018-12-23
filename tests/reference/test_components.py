from redblackgraph.reference import find_components, find_components_extended, topological_sort, transitive_closure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def test_find_components():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    A_star = transitive_closure(A).W
    components = find_components(A_star)
    assert components == [1, 2, 1, 1, 2, 1, 2]
    extended_components = find_components_extended(A_star)
    assert extended_components.ids == components
    assert len(extended_components.size_map) == 2
    assert extended_components.size_map[1] == 4
    assert extended_components.size_map[2] == 3


def test_find_components_dfs():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    A = coo_matrix((
        [-1, 2, 3, -1, 2, 1, -1, 2, -1, 3, 1, 1],
        (
            [0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 5, 6],
            [0, 3, 5, 1, 0, 2, 3, 1, 4, 6, 5, 6]
        )
    ))
    components = connected_components(A)
    print(components)
