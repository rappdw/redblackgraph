from typing import Sequence


def get_traversal_path(relationship: int) -> Sequence[str]:
    '''Given a number representing the relationship from a vertex u to a
    vertex v, return the traversal path of edges to red or black vertices
    to "walk" from the u to the v.

    For example, input of 14 results in ['b', 'b', 'r'] which indicates that starting at
    u, follow the edge to the black vertex, then the edge to the black
    vertex then the edge to the red vertex arriving at v.'''
    x = relationship
    path = []
    mask = 1
    while (x > 1):
        path.insert(0, 'b' if x & mask else 'r')
        x >>= 1
    return path
