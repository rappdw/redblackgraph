from typing import Sequence
from .components import find_components_extended
from .permutation import permute

from redblackgraph.reference.topological_sort import topological_sort
from ..types.ordering import Ordering


def avos_canonical_ordering(A: Sequence[Sequence[int]]) -> Ordering:
    """
    Canonically sort the matrix.

    This ordering is canonical. Graph components will appear in adjacent
    rows starting with the largest component in rows 0-n, the next largest in n+1-m, etc.
    Should the graph hold components of the same size, the component id will be used to
    order one above the other. Within a component, row ordering is determined first by
    maximum relationship value in a row and finally by original vertex id.

    :param A: input matrix (assumed to be transitively closed)
    :return: an upper triangular matrix that is isomorphic to A
    """

    ordering = find_components_extended(A).get_ordering()
    return Ordering(permute(A, ordering), ordering)

def topological_ordering(A: Sequence[Sequence[int]]) -> Ordering:
    """
    Relabel the graph so that the resultant Red Black adjacency matrix is upper triangular

    This ordering is not canonical, it is only guaranteed to produce an upper
    triangular representation. It does so by topologically sorting the graph (O(V+E)).

    Whereas canonical_sort assumes that the input matrix is transitively closed,
    this version does not.
    :param A: input red black adjacency matrix
    :return: an upper triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """

    # step 1: determine topological ordering of nodes in the graph
    ordering = topological_sort(A)
    # step 2: permute the matrix and return triangularization
    return Ordering(permute(A, ordering), ordering)
