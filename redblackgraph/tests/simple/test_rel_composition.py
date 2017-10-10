from redblackgraph.simple import relational_composition


def test_relational_composition():
    # use the A+ from the example in our notebook, add in a sibling to the last vertex in the graph
    A = [[-1, 2, 3, 4, 0],
         [0, -1, 0, 2, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, -1, 0],
         [2, 4, 5, 8, 1]]
    u = [[2, 0, 0, 0, 0, -1]]
    v = [[0], [0], [0], [0], [0], [-1]]
    A_lambda = relational_composition(u, A, v)
    assert A_lambda == [[-1, 2, 3, 4, 0, 0],
                        [0, -1, 0, 2, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0],
                        [2, 4, 5, 8, 1, 0],
                        [2, 4, 5, 8, 0, -1]]

    # Using the A_lambda that was generated... Add in the "great-grandmother" to the siblings represented by vertex 4 and 5
    u = [[0, 0, 0, 0, 0, 0, 1]]
    v = [[0], [3], [0], [0], [0], [0], [1]]
    A_lambda = relational_composition(u, A_lambda, v)
    assert A_lambda == [[-1, 2, 3, 4, 0, 0, 5],
                        [0, -1, 0, 2, 0, 0, 3],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0, 0],
                        [2, 4, 5, 8, 1, 0, 9],
                        [2, 4, 5, 8, 0, -1, 9],
                        [0, 0, 0, 0, 0, 0, 1]]
