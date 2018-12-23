import redblackgraph as rb

def test_loop():
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
        A_lambda = A_star.vertex_relational_composition(u, v, 1)
        print(A_lambda)
        assert False
    except ValueError:
        # expected
        pass
