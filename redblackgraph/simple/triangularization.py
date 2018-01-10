import numpy as np

from collections import defaultdict


def find_components_extended(A):
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
    u = [0] * len(A)
    v = [0] * len(A)
    q = defaultdict(lambda: 0)
    component_number = 1
    u[0] = component_number
    q[component_number] += 1
    for i in range(len(A)):
        row_max = -2
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
            q[component_number] += 1
        row_component_number = u[i]
        for j in range(len(A)):
            if A[i][j] != 0:
                row_max = max(A[i][j], row_max)
                if u[j] == 0:
                    u[j] = row_component_number
                    q[row_component_number] += 1
                elif u[j] != row_component_number:
                    # we've encountered an "overlapping" row, the row number really should
                    # be what we just found in u[j] and we'll need to reverse any changes
                    # we've made up to this point
                    for k in range(j):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                            q[row_component_number] -= 1
                            q[u[j]] += 1
                    if i > j:
                        u[i] = u[j]
                        q[u[j]] += 1
                        q[row_component_number] -= 1
                    component_number -= 1
                    row_component_number = u[j]
        v[i] = row_max
    return (u, v, {k:v for k,v in q.items() if v != 0})

def triangularize(A):
    """
    u, v, and q are computed via find_components_extended, and then used to compute a
    permutation matrix, P, and then return P @ A @ P
    :param A:
    :return: a triangular matrix that is symmetrical to A (a relabeling of the graph vertices)
    """
    u, v, q = find_components_extended(A)

    # this yeilds a list of tuples where each tuple is the size of the component, the component id of the vertex,
    # the max np for the vertex and the id of the vertex. We want the nodes
    # ordered by components size, componente id, max np, finally by vertex id
    permutation_basis = sorted(
        [(q[element[1][0]],) + element[1] + (element[0],) for element in enumerate(zip(u, v))],
        reverse=True
    )

    # from the permutation basis, create the permutation matrix
    n = len(permutation_basis)
    P = np.zeros(shape=(n, n), dtype=np.int32)
    for idx, element in enumerate(permutation_basis):
        P[idx][element[3]] = 1

    # triagularize A
    return (P @ A @ np.transpose(P)).tolist()
