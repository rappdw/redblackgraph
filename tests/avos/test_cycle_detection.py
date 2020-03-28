import redblackgraph as rb
import redblackgraph.reference as ref
import redblackgraph.sparse as sparse
from redblackgraph.util.capture import capture

import pytest

def core_transitive_closure(A):
    A = rb.array(A)
    return A.transitive_closure()

def test_relational_composition():
    # test simple avos matmul
    A = rb.array([[-1,  2,  3,  0,  0],
                  [ 0, -1,  0,  2,  0],
                  [ 0,  0,  1,  0,  0],
                  [ 0,  0,  0, -1,  0],
                  [ 2,  0,  0,  0,  1]])
    A_star = A.transitive_closure().W

    # what happens if we perform a relational_composition that causes a loop
    u = rb.array([0, 0, 0, 0, 3])
    v = rb.array([0, 0, 0, 3, 0])

    try:
        _ = A_star.vertex_relational_composition(u, v, 1)
        assert False
    except ValueError as e:
        assert str(e) == 'Relational composition would result in a cycle. Idx: 0, u_i: 6, v_i: 9'

@pytest.mark.parametrize(
    "transitive_closure",
    [
        (ref.transitive_closure),
        (core_transitive_closure),
        (sparse.transitive_closure_dijkstra),
        (sparse.transitive_closure_floyd_warshall),
    ]
)
def test_apsp_detection(transitive_closure):
    # test transitive closure loop detection
    A = [[-1,  2,  3,  0,  0],
         [ 0, -1,  0,  2,  0],
         [ 0,  0,  1,  0,  0],
         [ 0,  0,  0, -1,  3],
         [ 2,  0,  0,  0,  1]]
    try:
        B = transitive_closure(A).W
        print(capture(B))
        assert False
    except ValueError as e:
        assert 'Error: cycle detected!' in str(e)
