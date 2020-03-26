import numpy as np

from collections import defaultdict
from typing import Dict, List, Sequence
from .components import find_components
from .permutation import permute
from .rbg_math import MSB

from redblackgraph.reference.topological_sort import topological_sort
from redblackgraph.types.ordering import Ordering


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
        for j in vertices:
            if A[i][j]:
                ancester_count_for_vertex[i] += MSB(A[i][j])
            if A[j][i]:
                max_rel_for_vertex[i] = max(max_rel_for_vertex[i], A[j][i])

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
    perumutation =  _get_permutation(A, q, components)
    return Ordering(permute(A, perumutation), perumutation)

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
