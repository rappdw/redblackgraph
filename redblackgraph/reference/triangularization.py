import numpy as np

from dataclasses import dataclass
from typing import Dict, Sequence
from collections import defaultdict

@dataclass
class Components:
    ids: Sequence[int]
    max_pedigree_number: Sequence[int]
    size_map: Dict[int, int] # keyed by component id, valued by size of component

    def get_permutation_basis(self):
        # this yeilds a list of tuples where each tuple is the size of the component, the component id of the vertex,
        # the max np for the vertex and the id of the vertex. We want the nodes
        # ordered by components size, componente id, max np, finally by vertex id
        return sorted(
            [(self.size_map[element[1][0]],) + element[1] + (element[0],)
             for element in enumerate(zip(self.ids, self.max_pedigree_number))],
            reverse=True
        )

@dataclass
class Triangularization:
    A: Sequence[Sequence[int]]
    label_permutation: Sequence[int]

def find_components_extended(A: Sequence[Sequence[int]]) -> Components:
    """
    Given an input adjacency matrix (assumed to be transitively closed), triangularize the
    matrix (simply a relabeling of the graph)
    :param A: input adjacency matrix
    :return: a tuple of:
      [0] - a vector matching length of A with the elements holding the connected component id of
      the identified connected components - labeled u
      [1] - a vector matching length of A with the elements holding the max n_p for the corresponding
      row - labeled v
      [2] - a dictionary keyed by component id and valued by size of component
    """
    n = len(A)
    u = [0] * n
    v = [0] * n
    q = defaultdict(lambda: 0)
    component_number = 1
    u[0] = component_number
    q[component_number] += 1
    for i in range(n):
        row_max = -2
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
            q[component_number] += 1
        row_component_number = u[i]
        for j in range(n):
            if A[i][j] != 0:
                row_max = max(A[i][j], row_max)
                if u[j] == 0:
                    u[j] = row_component_number
                    q[row_component_number] += 1
                elif u[j] != row_component_number:
                    # There are a couple cases here. We implicitely assume a new row
                    # is a new component, so we need to back that out (iterate from 0
                    # to j), but we could also encounter a row that "merges" two
                    # components (need to sweep the entire u vector)
                    for k in range(n):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                            q[row_component_number] -= 1
                            q[u[j]] += 1
                    component_number -= 1
                    row_component_number = u[j]
        v[i] = row_max
    return Components(u, v, {k:v for k,v in q.items() if v != 0})

def _get_triangularization_permutation_matrices(A):
    """
    u, v, and q are computed via find_components_extended, and then used to compute a
    permutation matrix, P, and P_transpose
    :param A:
    :return: the permutation matrices that will triangularize A
    """
    permutation_basis = find_components_extended(A).get_permutation_basis()

    # from the permutation basis, create the permutation matrix
    n = len(permutation_basis)
    P = np.zeros(shape=(n, n), dtype=np.int32)
    P_transpose = np.zeros(shape=(n, n), dtype=np.int32)
    # label_permutation can be calculated as P @ np.arrange(n), but since we are running the index do it here
    label_permutation = np.arange(n)
    for idx, element in enumerate(permutation_basis):
        label_permutation[idx] = element[3]
        P[idx][element[3]] = 1
        P_transpose[element[3]][idx] = 1
    return P, P_transpose, label_permutation


def triangularize(A: Sequence[Sequence[int]]) -> Triangularization:
    """
    triangularize the matrix. Uses P and P_transpose if provided, otherwise computes
    the permutation matrices
    :param A:
    :param P: the transposition matrices (P and P_transpose)
    :return: a triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """
    P, P_t, label_permutation = _get_triangularization_permutation_matrices(A)

    # triagularize A
    return Triangularization((P @ A @ P_t), label_permutation)
