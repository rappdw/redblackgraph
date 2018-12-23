from typing import Sequence

def find_components(A: Sequence[Sequence[int]]) -> Sequence[int]:
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (transitively closed)
    :return: a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
    """
    n = len(A)
    u = [0] * n
    component_number = 1
    u[0] = component_number
    for i in range(n):
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
        row_component_number = u[i]
        for j in range(n):
            if A[i][j] != 0:
                if u[j] == 0:
                    u[j] = row_component_number
                elif u[j] != row_component_number:
                    # There are a couple cases here. We implicitely assume a new row
                    # is a new component, so we need to back that out (iterate from 0
                    # to j), but we could also encounter a row that "merges" two
                    # components (need to sweep the entire u vector)
                    for k in range(n):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                    component_number -= 1
                    row_component_number = u[j]
                    u[i] = row_component_number
    return u
