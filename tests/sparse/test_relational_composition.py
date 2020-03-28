from scipy.sparse import coo_matrix
from redblackgraph import rb_matrix, Color

def test_relational_composition():
    A = rb_matrix(coo_matrix(([0, 0, 2, 0, 1, 3, 1, 1, 3, 0], ([0, 1, 1, 2, 3, 3, 4, 5, 6, 6], [0, 1, 2, 2, 3, 5, 4, 5, 4, 6])), shape=(7, 7)).tocsr())
    u = rb_matrix(coo_matrix(([2, 3], ([0, 0], [1, 3])), shape=(1, 7)).tocsr())
    v = rb_matrix(coo_matrix(([2, 2], ([0, 4], [0, 0])), shape=(7, 1)).tocsr())

    # TODO: implement relational composition for sparse implementation
    # B = A.rc(u, v, Color.RED)
    # TODO: add assertions to validate the result

