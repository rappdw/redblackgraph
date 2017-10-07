from redblackgraph.simple import generation

def lookup_relationship(du, dv):
    removal = abs(du - dv)
    generational = min(du, dv)

    if generational == 1:
        assert removal == 0
        return "siblings"
    if generational == 2 and not removal == 0:
        return "aunt/uncle"
    if removal == 0:
        return f"{generational - 1} cousin"
    return f"{generational - 1} cousin {removal} removed"

def calculate_relationship(u, v):
    '''
    Determine if a relationship exists between u, v where u, v are row vectors of the transitive closure
    of a Red Black adjacency matrix
    :param u: row vector for vertex u
    :param v: row vector for vertex v
    :return: (Relationship designation, common ancestor vertex)
    '''

    common_ancestor, (x, y) = min([e for e in enumerate(zip(u, v)) if e[1][0] > 1 and e[1][1] > 1],
                                  key=lambda x: x[1][0] + x[1][1],
                                  default=(-1, (0, 0)))

    if common_ancestor == -1:
        return "No Relationship", -1
    return lookup_relationship(generation(x), generation(y)), common_ancestor
