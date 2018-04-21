from redblackgraph.reference import calculate_relationship, Relationship


def test_deirect_relationships():
    # from the sample adjacency matrix in the notebook, test calc_relationship(my daughter, me)
    u = [-1, 2, 3, 4, 0]
    v = [2, 4, 5, 8, 1]
    assert calculate_relationship(u, v) == Relationship(0, 'parent')

    # from the sample adjacency matrix in the notebook, test calc_relationship(my daughter, my father)
    u = [0, -1, 0, 2, 0]
    v = [2, 4, 5, 8, 1]
    assert calculate_relationship(u, v) == Relationship(1, 'grandparent')
