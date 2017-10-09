from redblackgraph.simple import calculate_relationship


def test_deirect_relationships():
    # from the sample adjacency matrix in the notebook, test calc_relationship(my daughter, me)
    u = [-1, 2, 3, 4, 0]
    v = [2, 4, 5, 8, 1]
    assert calculate_relationship(u, v) == ('-1 cousin 1 removed', 0) # TODO: this should be parent

    # from the sample adjacency matrix in the notebook, test calc_relationship(my daughter, my father)
    u = [0, -1, 0, 2, 0]
    v = [2, 4, 5, 8, 1]
    assert calculate_relationship(u, v) == ('-1 cousin 2 removed', 1) # TODO: this should be grandparent
