def generation(pedigree_number):
    '''Given a pedigree_number, representing a relationship from a "root" vertex to an "ancester" vertex,
    return the number of edges that must be followed in order to "walk" from the "root" to the "ancester".'''
    x = pedigree_number
    gen = 0
    while (x > 1):
        x >>= 1
        gen += 1
    return gen


def get_traversal_path(pedigree_number):
    '''Given a pedigree_number, representing a relationship from a "root" vertex to an "ancester" vertex,
    return the traversal path of edges to red or black vertices to "walk" from the "root" to the "ancesster".

    For example, input of 14 results in ['b', 'b', 'r'] which indicates that starting at the "root" vertex, follow
    the edge to the black vertex, then the edge to the black vertex then the edge to the red vertex.'''
    x = pedigree_number
    path = []
    mask = 1
    while (x > 1):
        path.insert(0, 'b') if x & mask else path.insert(0, 'r')
        x >>= 1
    return path
