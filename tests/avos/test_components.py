import redblackgraph.reference as ref
import redblackgraph.sparse.csgraph as sparse
import pytest

from numpy.testing import assert_equal
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

@pytest.mark.parametrize(
    "transitive_closure,find_components",
    [
        (ref.transitive_closure, ref.find_components),
        (sparse.transitive_closure_floyd_warshall, sparse.find_components),
        (sparse.transitive_closure_dijkstra, sparse.find_components)
    ]
)
def test_find_components(transitive_closure, find_components):
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

    assert_equal(A_star, expected_transitive_closure)

    components = find_components(A_star)
    assert_equal(components, [0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

@pytest.mark.parametrize(
    "transitive_closure,find_components",
    [
        (ref.transitive_closure, ref.find_components),
        (sparse.transitive_closure_floyd_warshall, sparse.find_components),
        (sparse.transitive_closure_dijkstra, sparse.find_components),
    ]
)
def test_find_components_use_case_2(transitive_closure, find_components):
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [  0, 0, 0, 0, 0, 0, 1]]
    A_star = transitive_closure(A).W
    expected_transitive_closure = [
        [-1, 0, 0, 2, 0, 3, 0],
        [ 0,-1, 0, 0, 0, 0, 0],
        [ 2, 0, 1, 4, 0, 5, 0],
        [ 0, 0, 0,-1, 0, 0, 0],
        [ 0, 2, 0, 0,-1, 0, 3],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 1]
    ]
    assert A_star.tolist() == expected_transitive_closure

    components = find_components(A_star)
    assert_equal(components, [0, 1, 0, 0, 1, 0, 1])

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
