from numpy.testing import assert_equal
from redblackgraph.reference import avos_canonical_ordering, topological_ordering, transitive_closure

def test_triangularize_via_topological_sort():
    A = [[-1, 0, 0, 2, 0, 3, 0],
         [ 0,-1, 0, 0, 0, 0, 0],
         [ 2, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0, 0, 0],
         [ 0, 2, 0, 0,-1, 0, 3],
         [ 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 1]]

    A_triangle = topological_ordering(A)
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
    A = [
      [-1, 0, 0, 2, 0, 3, 0],
      [ 0,-1, 0, 0, 0, 0, 0],
      [ 2, 0, 1, 0, 0, 0, 0],
      [ 0, 0, 0,-1, 0, 0, 0],
      [ 0, 2, 0, 0,-1, 0, 3],
      [ 0, 0, 0, 0, 0, 1, 0],
      [ 0, 0, 0, 0, 0, 0, 1]
    ]
    A_star = transitive_closure(A).W

    expected_A_star = [
      [-1, 0, 0, 2, 0, 3, 0],
      [ 0,-1, 0, 0, 0, 0, 0],
      [ 2, 0, 1, 4, 0, 5, 0],
      [ 0, 0, 0,-1, 0, 0, 0],
      [ 0, 2, 0, 0,-1, 0, 3],
      [ 0, 0, 0, 0, 0, 1, 0],
      [ 0, 0, 0, 0, 0, 0, 1]
    ]
    assert_equal(A_star, expected_A_star)

    A_star_canonical = avos_canonical_ordering(A_star)
    expected_canonical = [[ 1, 2, 4, 5, 0, 0, 0],
                          [ 0,-1, 2, 3, 0, 0, 0],
                          [ 0, 0,-1, 0, 0, 0, 0],
                          [ 0, 0, 0, 1, 0, 0, 0],
                          [ 0, 0, 0, 0,-1, 2, 3],
                          [ 0, 0, 0, 0, 0,-1, 0],
                          [ 0, 0, 0, 0, 0, 0, 1]]

#    print(capture(A_star_canonical.A))
    assert_equal(A_star_canonical.A, expected_canonical)
    assert_equal(A_star_canonical.label_permutation, [2, 0, 3, 5, 4, 1, 6])
