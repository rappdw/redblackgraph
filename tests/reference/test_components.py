from redblackgraph.reference import find_components, find_components_extended, transitive_closure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def test_find_components():
    A = [
        [-1, 0, 0, 0, 3, 2, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 3,-1, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 1, 0, 0, 0, 2, 3, 0],
        [ 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 2],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    ]
    A_star = transitive_closure(A).W

    expected_transitive_closure = [
        [-1, 0, 0, 0, 3, 2, 7, 0, 0, 4],
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 2, 3,-1, 0, 5, 4,11, 0, 0, 8],
        [ 0, 0, 0, 1, 0, 0, 0, 2, 3, 0],
        [ 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 2],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    ]

    assert A_star.tolist() == expected_transitive_closure

    components = find_components(A_star)
    assert components == [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]

    extended_components = find_components_extended(A_star)
    assert extended_components.ids == components
    assert len(extended_components.size_map) == 2
    assert extended_components.size_map[0] == 7
    assert extended_components.size_map[1] == 3


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
    assert components[0] == 2
    assert components[1].tolist() == [0, 1, 0, 0, 1, 0, 1]
