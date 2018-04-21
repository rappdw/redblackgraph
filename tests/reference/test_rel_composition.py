from redblackgraph.reference import vertex_relational_composition, edge_relational_composition, warshall

def test_vertex_relational_composition():
    # use the A+ from the example in our notebook, add in a sibling to the last vertex in the graph
    A = [[-1, 2, 3, 4, 0],
         [0, -1, 0, 2, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, -1, 0],
         [2, 4, 5, 8, 1]]
    u = [[2, 0, 0, 0, 0]]
    v = [[0], [0], [0], [0], [0]]
    A_lambda = vertex_relational_composition(u, A, v, -1)
    assert A_lambda == [[-1, 2, 3, 4, 0, 0],
                        [0, -1, 0, 2, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0],
                        [2, 4, 5, 8, 1, 0],
                        [2, 4, 5, 8, 0, -1]]

    # Using the A_lambda that was generated... Add in the "great-grandmother" to the siblings represented by vertex 4 and 5
    u = [[0, 0, 0, 0, 0, 0]]
    v = [[0], [3], [0], [0], [0], [0]]
    A_lambda = vertex_relational_composition(u, A_lambda, v, 1)
    assert A_lambda == [[-1, 2, 3, 4, 0, 0, 5],
                        [0, -1, 0, 2, 0, 0, 3],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0, 0, 0],
                        [2, 4, 5, 8, 1, 0, 9],
                        [2, 4, 5, 8, 0, -1, 9],
                        [0, 0, 0, 0, 0, 0, 1]]

def test_my_use_case_vertex():
    #        D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
    A1 = [[ -1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
          [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0],  # E
          [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],  # R
          [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
          [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0],  # Mi
          [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0],  # Do
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # Ev
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],  # G
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # Ma
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],  # S
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]  # Em
         ]
    #      D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
    u = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3]]
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
         ]
    R = warshall(A1)[0].tolist()
    A_lambda = vertex_relational_composition(u, R, v, -1)
    assert A_lambda[0][12] == 12
    assert A_lambda[0][13] == 13
    assert A_lambda[0][14] == 6
    assert A_lambda[2][12] == 4
    assert A_lambda[2][13] == 5
    assert A_lambda[2][14] == 2
    assert A_lambda[8][12] == 4
    assert A_lambda[8][13] == 5
    assert A_lambda[8][14] == 2
    # print()
    # print(np.array(A_lambda))

def test_edge_relational_composition_simple():
    R = [[-1, 0, 3, 0, 0],
         [ 0,-1, 0, 2, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0,-1, 0],
         [ 2, 0, 5, 0, 1]]
    R_lambda = edge_relational_composition(R, 0, 1, 2)
    R_expected = [[-1, 2, 3, 4, 0],
                  [ 0,-1, 0, 2, 0],
                  [ 0, 0, 1, 0, 0],
                  [ 0, 0, 0,-1, 0],
                  [ 2, 4, 5, 8, 1]]
    assert R_lambda == R_expected

def test_my_use_case_edge():
    #        D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
    A1 = [[ -1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # D
          [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],  # E
          [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],  # R
          [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # H
          [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
          [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0,  2],  # Do
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0], # Em
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]  # J
          ]
    R = warshall(A1)[0]
    # Missing edge is R -> J, 2
    A_lambda = edge_relational_composition(R, 2, 14, 2)
    assert A_lambda[0][12] == 12
    assert A_lambda[0][13] == 13
    assert A_lambda[0][14] == 6
    assert A_lambda[2][12] == 4
    assert A_lambda[2][13] == 5
    assert A_lambda[2][14] == 2
    assert A_lambda[8][12] == 4
    assert A_lambda[8][13] == 5
    assert A_lambda[8][14] == 2
