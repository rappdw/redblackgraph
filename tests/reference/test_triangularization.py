from numpy.testing import assert_equal
from redblackgraph.reference import canonical_sort, triangularize, transitive_closure

def test_triangularize_via_topological_sort():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]

    A_triangle = triangularize(A)
    assert_equal(A_triangle.label_permutation, [4, 6, 2, 1, 0, 3, 5])

    expected = [[-1, 3, 0, 2, 0, 0, 0],
                [ 0, 1, 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 2, 0, 0],
                [ 0, 0, 0,-1, 0, 0, 0],
                [ 0, 0, 0, 0,-1, 2, 3],
                [ 0, 0, 0, 0, 0,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 1]]
    assert_equal(A_triangle.A, expected)

def test_triangularization():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]
    A_star = transitive_closure(A).W
    A_star_canonical = canonical_sort(A_star)
    expected_canonical = [[ 1, 2, 5, 4, 0, 0, 0],
                          [ 0,-1, 3, 2, 0, 0, 0],
                          [ 0, 0, 1, 0, 0, 0, 0],
                          [ 0, 0, 0,-1, 0, 0, 0],
                          [ 0, 0, 0, 0,-1, 3, 2],
                          [ 0, 0, 0, 0, 0, 1, 0],
                          [ 0, 0, 0, 0, 0, 0,-1]]

    assert_equal(A_star_canonical.A, expected_canonical)
    assert_equal(A_star_canonical.label_permutation, [2, 0, 5, 3, 4, 6, 1])

    #TODO: add test case to ensure we've fixed the "row merges two components" use case