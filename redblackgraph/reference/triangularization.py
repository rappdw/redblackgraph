import numpy as np

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List
from collections import defaultdict
from .util import MSB

from redblackgraph.reference.topological_sort import topological_sort

@dataclass
class Components:
    ids: Sequence[int]
    rel_count: Sequence[int]
    max_rel: Sequence[int]
    size_map: Dict[int, int] # keyed by component id, valued by size of component

    def get_permutation_basis(self) -> List[Tuple[int, int, int, int, int]]:
        # This yields a list of tuples. Every vertex is represented in this list and each tuple is:
        #   - the size of the component
        #   - the component id of the vertex
        #   - count(rel(u,v)) for the vertex
        #   - max(rel(u,v)) for the vertex
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

    # see algorithm description in .components find_components

    n = len(A)
    component_for_vertex = [0] * n
    max_rel_for_vertex = [0] * n
    ancester_count_for_vertex = [0] * n
    component_number = 0
    for i in range(n):
        assumed_new_component = False
        if component_for_vertex[i] == 0:
            component_number += 1
            component_for_vertex[i] = component_number
            assumed_new_component = True
        assumed_component_number = component_for_vertex[i]
        actual_new_component = True
        spanned_components = set()
        spanned_components.add(assumed_component_number)
        for j in range(n):
            if i != j and A[i][j] != 0:
                ancester_count_for_vertex[i] += MSB(A[i][j])
                max_rel_for_vertex[j] = max(A[i][j], max_rel_for_vertex[j])
                if component_for_vertex[j] != 0:
                    spanned_components.add(component_for_vertex[j])
                    actual_new_component = False
                component_for_vertex[j] = assumed_component_number
        if not actual_new_component:
            root_component = min(spanned_components)
            spanned_components.remove(root_component)
            for j in range(n):
                if component_for_vertex[j] in spanned_components:
                    component_for_vertex[j] = root_component
            if assumed_new_component:
                component_number -= 1
    q = defaultdict(lambda: 0)
    for i in range(n):
        q[component_for_vertex[i]] += 1
    return Components(component_for_vertex, ancester_count_for_vertex, max_rel_for_vertex, {k:v for k,v in q.items() if v != 0})

def _get_triangularization_permutation_matrices(A):
    """
    u, v, and q are computed via find_components_extended, and then used to compute a
    permutation matrix, P, and P_transpose
    :param A:
    :return: the permutation matrices that will canonical_sort A
    """
    components = find_components_extended(A)
    permutation_basis = components.get_permutation_basis()

    # from the permutation basis, create the permutation matrix
    n = len(permutation_basis)
    P = np.zeros(shape=(n, n), dtype=np.int32)
    P_transpose = np.zeros(shape=(n, n), dtype=np.int32)
    # label_permutation can be calculated as P @ np.arrange(n), but since we are running the index do it here
    label_permutation = np.arange(n)
    for idx, element in enumerate(permutation_basis):
        label_permutation[idx] = element[-1]
        P[idx][element[-1]] = 1
        P_transpose[element[-1]][idx] = 1
    return P, P_transpose, label_permutation


def canonical_sort(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    Canonically sort the matrix.

    This form of triangularization is canonical. Graph components will appear in adjacent
    rows starting with the largest component in rows 0-n, the next largest in n+1-m, etc.
    Should the graph hold components of the same size, the component id will be used to
    order one above the other. Within a component, row ordering is determined first by
    maximum relationship value in a row and finally by original vertex id.

    This is an expensive operation. First it assumes that A is transitively closed (O(n^3)).
    It then computes the components of the graph (O(n^3)). It then sorts the resultant
    component information (O(n logn)). Based on this it computes permutation matrices (O(n))
    and finally uses the permutation matrices to reorder the graph (O(n^2))

    :param A: input matrix (assumed to be transitively closed)
    :param P: the transposition matrices (P and P_transpose)
    :return: an upper triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """

    P, P_t, label_permutation = _get_triangularization_permutation_matrices(A)

    # triagularize A
    return Triangularization((P @ A @ P_t), label_permutation)

def triangularize(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    Relabel the graph so that the resultant Red Black adjacency matrix is upper triangular

    This form of triangularization is not canonical, it is only guaranteed to produce an upper
    triangular representation. It does so by topologically sorting the graph (O(V+E)). Then
    producing permutation matrices (O(n)). Finally using the permutation matrices to reorder the
    graph (O(n^2)).

    Whereas canonical_sort assumes that the input matrix is transitively closed,
    this version does not.
    :param A: input red black adjacency matrix
    :return: an upper triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """

    # step 1: determine topological ordering of nodes in the graph
    n = len(A)
    ordering = topological_sort(A)
    # step 2: setup the permutation matrix
    P = np.zeros(shape=(n, n), dtype=np.int32)
    for idx, i in enumerate(ordering):
        P[idx][i] = 1
    # step 3: permute the matrix and return triangularization
    return Triangularization(P @ A @ P.T, ordering)
