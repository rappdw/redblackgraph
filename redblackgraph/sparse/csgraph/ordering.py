from typing import Sequence
from redblackgraph.types.ordering import Ordering
from ._components import find_components_extended
from ._permutation import permute

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
