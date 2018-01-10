from redblackgraph.simple import find_components, find_components_extended, triangularize, warshall


def test_find_components():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    A_star = warshall(A)[0]
    components = find_components(A_star)
    assert components == [1, 2, 1, 1, 2, 1, 2]
    extended_components = find_components_extended(A_star)
    assert extended_components[0] == components
    assert len(extended_components[2]) == 2
    assert extended_components[2][1] == 4
    assert extended_components[2][2] == 3

    A_star_canonical = triangularize(A_star)
    expected_canonical = [[ 1, 2, 5, 4, 0, 0, 0],
                          [ 0,-1, 3, 2, 0, 0, 0],
                          [ 0, 0, 1, 0, 0, 0, 0],
                          [ 0, 0, 0,-1, 0, 0, 0],
                          [ 0, 0, 0, 0,-1, 3, 2],
                          [ 0, 0, 0, 0, 0, 1, 0],
                          [ 0, 0, 0, 0, 0, 0,-1]]

    assert A_star_canonical == expected_canonical
