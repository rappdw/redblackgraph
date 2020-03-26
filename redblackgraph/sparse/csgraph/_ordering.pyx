import numpy as np
cimport numpy as np

import itertools as it

from collections import defaultdict
from typing import Dict, List, Sequence
from redblackgraph.types.ordering import Ordering
from ._components import find_components
from ._permutation import permute

include 'parameters.pxi'
include '_rbg_math.pxi'

def _get_permutation(A: Sequence[Sequence[int]], q: Dict[int, int], ids: Sequence[int]) -> List[int]:
    # This is the default sort ordering used by Traingularization
    # it sorts by:
    #   * size of component, descending
    #   * component id, ascending
    #   * relationship count, descending
    #   * max ancestor: ascending
    #   * color: descending
    #   * vertex_id: ascending
    n = len(A)
    max_rel_for_vertex = np.zeros((n), dtype=np.int32)
    ancester_count_for_vertex = np.zeros((n), dtype=np.int32)
    vertices = range(n)
    for i in vertices:
        for j in it.filterfalse(lambda x: (A[i][x] == 0 and A[x][i] == 0) or x == i, vertices):
            max_rel_for_vertex[i] = max(max_rel_for_vertex[i], A[j][i])
            ancester_count_for_vertex[i] += MSB(A[i][j])

    basis = [i for i in range(len(ids))]
    # sort descending on size of component and "ancestor count", ascending on all other elements
    basis.sort(key=lambda x: (-q[ids[x]], ids[x], -ancester_count_for_vertex[x], max_rel_for_vertex[x], -A[x][x], x))
    return basis


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

    q = defaultdict(lambda: 0) # dictionary keyed by component id, value is count of vertices in componenet
    components = find_components(A, q)
    perumutation = np.array(_get_permutation(A, q, components), dtype=ITYPE)
    return Ordering(permute(A, perumutation), perumutation)
