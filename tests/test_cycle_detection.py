import redblackgraph as rb

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

def test_reference_impl():
    # test transitive closure loop detection
    A = [[-1,  2,  3,  0,  0],
         [ 0, -1,  0,  2,  0],
         [ 0,  0,  1,  0,  0],
         [ 0,  0,  0, -1,  3],
         [ 2,  0,  0,  0,  1]]
    try:
        _ = rb.reference.transitive_closure(A).W
        assert False
    except ValueError as e:
        assert str(e) == 'Error: cycle detected! Vertex 4 has a path to itself. A(4,3)=8, A(3,4)=3'

def test_numpy_impl():
    # test transitive closure loop detection
    A = rb.array([[-1,  2,  3,  0,  0],
                  [ 0, -1,  0,  2,  0],
                  [ 0,  0,  1,  0,  0],
                  [ 0,  0,  0, -1,  3],
                  [ 2,  0,  0,  0,  1]])
    try:
        _ = A.transitive_closure().W
        assert False
    except ValueError as e:
        assert str(e) == 'Error: cycle detected! Vertex 3 has a path to itself. A(3,4)=3, A(4,3)=8'

