from redblackgraph.simple import generation


def lookup_relationship(du, dv):
    '''
    This is a very rudimentary implementation of a Consanguinity lookup and doesn't handle many cases
    correctly.
    :param du: generational distance from u to common ancestor
    :param dv: generational distance from v to common ancester
    :return: a string designating relationship
    '''
    removal = abs(du - dv)
    generational = min(du, dv)
    return f"{generational - 1} cousin {removal} removed"


def calculate_relationship(u, v):
    '''
    Determine if a relationship exists between u, v where u, v are row vectors of the transitive closure
    of a Red Black adjacency matrix
    :param u: row vector for vertex u
    :param v: row vector for vertex v
    :return: (Relationship designation, common ancestor vertex)
    '''

    common_ancestor, (x, y) = min([e for e in enumerate(zip(u, v)) if not e[1][0] == 0 and not e[1][1] == 0],
                                  key=lambda x: x[1][0] + x[1][1],
                                  default=(-1, (0, 0)))

    if common_ancestor == -1:
        return "No Relationship", -1
    return lookup_relationship(generation(x), generation(y)), common_ancestor
