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

def test_my_use_case():
    #       D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
    A1 = [[ -1,  2,  3, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
          [  0, -1,  0, 0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0],  # E
          [  0,  0,  1, 0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],  # R
          [  0,  0,  0, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0, 3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
          [  0,  0,  0, 0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0],  # Mi
          [  0,  0,  0, 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0, 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0, 0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0],  # Do
          [  0,  0,  0, 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # Ev
          [  0,  0,  0, 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],  # G
          [  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # Ma
          [  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],  # S
          [  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
         ]
    #      D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
    u = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]]
    v = [[ 0], # D
     [ 0], # E
     [ 2], # R
     [ 0], # M
     [ 0], # H
     [ 0], # Mi
     [ 0], # A
     [ 0], # I
     [ 2], # Do
     [ 0], # Ev
     [ 0], # G
     [ 0], # Ma
     [ 0], # S
     [ 0], # Em
     [-1]] # J
    A_lambda = relational_composition(u, A1, v)
    assert A_lambda[0][12] == 12
    assert A_lambda[0][13] == 13
    assert A_lambda[0][14] == 6
    assert A_lambda[2][12] == 4
    assert A_lambda[2][13] == 5
    assert A_lambda[2][14] == 2
    assert A_lambda[8][12] == 4
    assert A_lambda[8][13] == 5
    assert A_lambda[8][14] == 2

