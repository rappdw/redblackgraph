import numpy as np
cimport numpy as np
cimport cython

from typing import Dict, List, Sequence
from redblackgraph.types.ordering import Ordering
from ._components import find_components
from ._permutation import permute

include 'parameters.pxi'
include '_rbg_math.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_permutation(A: Sequence[Sequence[int]], q: Dict[int, int], ids: Sequence[int]) -> List[int]:
    # This is the default sort ordering used by Traingularization
    # it sorts by:
    #   * size of component, descending
    #   * component id, ascending
    #   * relationship count, descending
    #   * max ancestor: ascending
    #   * color: descending
    #   * vertex_id: ascending
    cdef unsigned int n = len(A)
    cdef DTYPE_t[:] max_rel_for_vertex = np.zeros((n), dtype=np.int32)
    cdef DTYPE_t[:] ancester_count_for_vertex = np.zeros((n), dtype=np.int32)
    cdef DTYPE_t[:, :] Am = A
    vertices = range(n)
    for i in vertices:
        for j in vertices:
            if Am[i][j]:
                ancester_count_for_vertex[i] += MSB(Am[i][j])
            if Am[j][i]:
                max_rel_for_vertex[i] = max(max_rel_for_vertex[i], Am[j][i])

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

    q = dict() # dictionary keyed by component id, value is count of vertices in component
    components = find_components(A, q)
    permutation = np.array(_get_permutation(A, q, components), dtype=ITYPE)
    return Ordering(permute(A, permutation), permutation, q)
