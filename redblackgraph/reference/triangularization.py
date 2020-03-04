import itertools as it

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List
from collections import defaultdict
from .components import find_components
from .util import MSB
from .permutation import permute
from ..util.capture import capture

from redblackgraph.reference.topological_sort import topological_sort

@dataclass
class Components:
    ids: Sequence[int]
    rel_count: Sequence[int]
    max_rel: Sequence[int]
    size_map: Dict[int, int] # keyed by component id, valued by size of component

    def get_ordering(self) -> List[Tuple[int, int, int, int, int]]:
        # This yields a list of tuples. Every vertex is represented in this list and each tuple is:
        #   - the size of the component
        #   - the component id of the vertex
        #   - count(rel(u,v)) for the vertex (row operation)
        #   - max(rel(u,v)) for the vertex (column operation)
        #   - the vertex id
        # This is the default sort ordering used by Traingularization
        basis = [(self.size_map[element[1][0]],) + element[1] + (element[0],) for element in
                         enumerate(zip(self.ids, self.rel_count, self.max_rel))]
        # sort descending on size of component and "ancestor count", asecending on all other elements
        return sorted(basis, key=lambda element: (-element[0], element[1], -element[2], element[3], element[4]))

@dataclass
class Triangularization:
    A: Sequence[Sequence[int]]
    label_permutation: Sequence[int]

def find_components_extended(A: Sequence[Sequence[int]]) -> Components:
    """
    Given an input adjacency matrix (assumed to be transitively closed), find the distinct
    graph components
    :param A: input adjacency matrix
    :return: a tuple of:
      [0] - a vector matching length of A with the elements holding the connected component id of
      the identified connected components
      [1] - a vector matching length of A with the elements holding the ancestor position for the corresponding
      column
      [2] - a vector matching length of A with a count of ancestors for the corresponding row
      [3] - a dictionary keyed by component id and valued by size of component
    """
    # 60 seconds execution time (3633 vertices)
    # q = defaultdict(lambda: 0)
    # component_for_vertex = find_components(A, q)
    #
    # vertices = range(len(A))
    # max_rel_for_vertex = [reduce(max, [A[j][i] for j in it.filterfalse(lambda x: (A[i][x] == 0 and A[x][i] == 0) or x == i, vertices)]) for i in vertices]
    #
    # return Components(component_for_vertex, max_rel_for_vertex, {k:v for k,v in q.items() if v != 0})

    q = defaultdict(lambda: 0)
    component_for_vertex = find_components(A, q)

    # 70 seconds exeuction time (3633 vertices)
    n = len(A)
    max_rel_for_vertex = [0] * n
    ancester_count_for_vertex = [0] * n
    vertices = range(n)
    # if this loop is removed, execution time is: 17 seconds (3633 vertices)
    for i in vertices:
        for j in it.filterfalse(lambda x: (A[i][x] == 0 and A[x][i] == 0) or x == i, vertices):
            max_rel_for_vertex[i] = max(max_rel_for_vertex[i], A[j][i])
            ancester_count_for_vertex[i] += MSB(A[i][j])

    return Components(component_for_vertex, ancester_count_for_vertex, max_rel_for_vertex, {k:v for k,v in q.items() if v != 0})

def _get_triangularization_ordering(A: Sequence[Sequence[int]]) -> Sequence[int]:
    """
    Assumes that A is transitively closed. Finds the components and returns the vertex ordering
    based on the extended component information
    :param A: transitively closed RB Graph
    :return: ordering
    """
    components = find_components_extended(A)
    ordering = [element[-1] for element in components.get_ordering()]
    return ordering

def canonical_sort(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    Canonically sort the matrix.

    This form of triangularization is canonical. Graph components will appear in adjacent
    rows starting with the largest component in rows 0-n, the next largest in n+1-m, etc.
    Should the graph hold components of the same size, the component id will be used to
    order one above the other. Within a component, row ordering is determined first by
    maximum relationship value in a row and finally by original vertex id.

    :param A: input matrix (assumed to be transitively closed)
    :return: an upper triangular matrix that is isomorphic to A
    """

    ordering = _get_triangularization_ordering(A)
    return Triangularization(permute(A, ordering), ordering)

def triangularize(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    Relabel the graph so that the resultant Red Black adjacency matrix is upper triangular

    This form of triangularization is not canonical, it is only guaranteed to produce an upper
    triangular representation. It does so by topologically sorting the graph (O(V+E)).

    Whereas canonical_sort assumes that the input matrix is transitively closed,
    this version does not.
    :param A: input red black adjacency matrix
    :return: an upper triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """

    # step 1: determine topological ordering of nodes in the graph
    ordering = topological_sort(A)
    # step 2: permute the matrix and return triangularization
    return Triangularization(permute(A, ordering), ordering)
