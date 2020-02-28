from typing import Sequence

def find_components(A: Sequence[Sequence[int]]) -> Sequence[int]:
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (transitively closed)
    :return: a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
    """

    # Component identification is usually done using iterative dfs for each vertex. Since we have
    # a transitively closed RBG, we have implicit DFS info in each row. This algorithm utilizes that
    # fact.
    #
    # This is our algorithm:
    #
    # Allocate an array that will represent the component for each vertex
    # Iterate over each row:
    #   We assume that each row is a new component
    #   In a given row, iterate over the columns:
    #     Any non-zero columns in that row will be assigned to the row component if they are not already
    #     assigned a component
    #
    #     If they are already assigned a component, then our assumption that the row was a new
    #     component is invalid. This row actually belongs to the component indicated by the row
    #     component of the current column. There are 2 cases here:
    #       Case 1: Only our assumption of a new component for this row is invalid.
    #               In this case the actual component is the component already assigned to that column
    #       Case 2: This row spans 2 existing components
    #               In this case the actual component is the lowest component number of spanned components
    #
    #     Identify both cases by keeping a set for each row of actual components. Once the row iteration
    #     is complete simply iterate over the component vector changing all component assignments for the
    #     falsely assumed row component (and any merged components) to the actual component
    #

    n = len(A)
    component_for_vertex = [0] * n
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
    return component_for_vertex
