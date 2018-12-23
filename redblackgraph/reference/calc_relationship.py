from typing import Sequence
from redblackgraph.types.relationship import Relationship
from redblackgraph.reference.util import MSB


def lookup_relationship(da: int, db: int) -> str:
    '''
    This is a very rudimentary implementation of a Consanguinity lookup and doesn't handle many
    cases correctly.
    :param da: generational distance from u to common ancestor
    :param db: generational distance from v to common ancester
    :return: a string designating relationship
    '''
    removal = abs(da - db)
    if da == 0 or db == 0:
        # direct ancestor
        if removal == 1:
            return "parent"
        if removal == 2:
            return "grandparent"
        if removal == 3:
            return "great grandparent"
        return f"{removal - 2} great grandparent"
    else:
        # cousin
        generational = min(da, db)
        return f"{generational - 1} cousin {removal} removed"


def calculate_relationship(a: Sequence[int], b: Sequence[int]) -> Relationship:
    '''
    Determine if a relationship exists between u, v where u, v are row vectors of the transitive
    closure of a Red Black adjacency matrix
    :param a: row vector for vertex u
    :param b: row vector for vertex v
    :return: (Relationship designation, common ancestor vertex)
    '''

    common_ancestor, (x, y) = min([e for e in enumerate(zip(a, b))
                                   if not e[1][0] == 0 and not e[1][1] == 0],
                                  key=lambda x: x[1][0] + x[1][1],
                                  default=(-1, (0, 0)))

    if common_ancestor == -1:
        return Relationship(-1, "No Relationship")
    return Relationship(common_ancestor,
                        lookup_relationship(MSB(x), MSB(y)))
