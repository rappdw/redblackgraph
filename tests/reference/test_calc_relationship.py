from redblackgraph.reference import calculate_relationship, Relationship


def test_deirect_relationships():
    # my daughter, me
    u = [-1, 0]
    v = [ 2, 1]
    assert calculate_relationship(u, v) == Relationship(0, 'parent')

    # my daughter, my father
    u = [-1, 0]
    v = [ 4, 1]
    assert calculate_relationship(u, v) == Relationship(0, 'grandparent')

    # my son, my maternal grandmother
    u = [ 1,  0]
    v = [ 11,-1]
    assert calculate_relationship(u, v) == Relationship(0, 'great grandparent')

    # 2nd great grandparent
    u = [ 1,  0]
    v = [ 16,-1]
    assert calculate_relationship(u, v) == Relationship(0, '2nd great grandparent')

    # 3rd great grandparent
    u = [ 1,  0]
    v = [ 32,-1]
    assert calculate_relationship(u, v) == Relationship(0, '3rd great grandparent')

    # 4th great grandparent
    u = [ 1,  0]
    v = [ 64,-1]
    assert calculate_relationship(u, v) == Relationship(0, '4th great grandparent')

    # myself, my brother
    u = [-1,  0, 2]
    v = [ 0, -1, 2]
    assert calculate_relationship(u, v) == Relationship(2, 'sibling')

    # myself, my uncle
    u = [-1,  0, 4]
    v = [ 0, -1, 2]
    assert calculate_relationship(u, v) == Relationship(2, 'aunt/uncle')

    # myself, my cousin
    u = [-1,  0, 4]
    v = [ 0, -1, 4]
    assert calculate_relationship(u, v) == Relationship(2, '1st cousin')

    # myself, my 2nd cousin
    u = [-1,  0, 8]
    v = [ 0, -1, 8]
    assert calculate_relationship(u, v) == Relationship(2, '2nd cousin')

    # myself, my 3rd cousin
    u = [-1,  0, 16]
    v = [ 0, -1, 16]
    assert calculate_relationship(u, v) == Relationship(2, '3rd cousin')

    # myself, my 4th cousin
    u = [-1,  0, 32]
    v = [ 0, -1, 32]
    assert calculate_relationship(u, v) == Relationship(2, '4th cousin')

    # myself, my cousin once remove
    u = [-1,  0, 4]
    v = [ 0, -1, 8]
    assert calculate_relationship(u, v) == Relationship(2, '1st cousin 1 removed')

    # myself, my 2nd cousin
    u = [-1,  0, 8]
    v = [ 0, -1, 16]
    assert calculate_relationship(u, v) == Relationship(2, '2nd cousin 1 removed')

    # myself, my 3rd cousin
    u = [-1,  0, 16]
    v = [ 0, -1, 32]
    assert calculate_relationship(u, v) == Relationship(2, '3rd cousin 1 removed')

    # myself, my 4th cousin
    u = [-1,  0, 32]
    v = [ 0, -1, 64]
    assert calculate_relationship(u, v) == Relationship(2, '4th cousin 1 removed')

    # no relationship
    u = [-1,  0]
    v = [ 0, -1]
    assert calculate_relationship(u, v) == Relationship(-1, 'No Relationship')
