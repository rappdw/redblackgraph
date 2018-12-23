from redblackgraph.reference import vec_avos


def test_identity():
    # from the sample adjacency matrix in the notebook, test vec_avos(me, father)
    u = [-1, 2, 3, 0, 0]
    v = [2, -1, 0, 0, 0]
    assert vec_avos(u, v) == 2

    # from the sample adjacency matrix in the notebook, test vec_avos(me, mother)
    u = [-1, 2, 3, 0, 0]
    v = [3, 0, 1, 0, 0]
    assert vec_avos(u, v) == 3


def test_transitive():
    # from the sample adjacency matrix in the notebook, test vec_avos(my daughter, my father)
    u = [2, 0, 0, 0, 1]
    v = [2, -1, 0, 0, 0]
    assert vec_avos(u, v) == 4

