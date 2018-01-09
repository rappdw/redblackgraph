def find_components(A):
    """
    Given an input adjacency matrix (assumed to be transitively closed), compute the connected
    components
    :param A: input adjacency matrix
    :return: a tuple of:
      [0] - a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
      [1] - the number of connected components identified
    """
    u = [0] * len(A)
    component_number = 1
    u[0] = component_number
    for i in range(len(A)):
        if u[i] == 0:
            component_number += 1
            u[i] = component_number
        row_component_number = u[i]
        for j in range(len(A)):
            if A[i][j] != 0:
                if u[j] == 0:
                    u[j] = row_component_number
                elif u[j] != row_component_number:
                    # we've encountered an "overlapping" row, the row number really should
                    # be what we just found in u[j] and we'll need to reverse any changes
                    # we've made up to this point
                    for k in range(j):
                        if u[k] == row_component_number:
                            u[k] = u[j]
                    component_number -= 1
                    row_component_number = u[j]
                    u[i] = row_component_number
    return (u, component_number)
