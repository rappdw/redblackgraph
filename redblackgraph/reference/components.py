import itertools as it
from typing import Sequence

def find_components(A: Sequence[Sequence[int]]) -> Sequence[int]:
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (this implementation assumes that it is transitively closed)
    :return: a vector with matching length of A with the elements holding the connected component id of
    the identified connected components
    """

    # Component identification is usually done using iterative dfs for each vertex. Since A is
    # transitively closed, we have implicit DFS info in each row. This algorithm utilizes that
    # fact. Conceptually, this algorithm "crawls" the matrix.
    #
    # This is our algorithm:
    #
    # Allocate an array that will represent the component for each vertex
    # Allocate a set that contains the vertices visited
    # Iterate over each row not in the visited vertices:
    #   This is a new component, so increment the component id and assign the vertex of the current row to that id
    #   Allocate a set that holds vertices that will be added to this component
    #   In a given row, iterate over the columns not in the visited vertices (since this is a new component,
    #   no columns will be in the visited vertices):
    #     Any non-zero columns in that row will be assigned to the row component and added to the set of added vertices
    #

    n = len(A)
    component_for_vertex = [0] * n
    vertices = range(n)
    visited_vertices = set()
    component_id = 0
    for i in it.filterfalse(lambda x: x in visited_vertices, vertices):
        vertices_added_to_component = set()
        vertices_added_to_component.add(i)
        while vertices_added_to_component:
            vertex = vertices_added_to_component.pop()
            visited_vertices.add(vertex)
            component_for_vertex[vertex] = component_id
            for j in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[vertex][x] == 0 or x == vertex, vertices):
                vertices_added_to_component.add(j)
                for k in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[x][j] == 0 or x == j, vertices):
                    vertices_added_to_component.add(k)
            # now we need to iterate the vertex's column
            for k in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[x][vertex] == 0 or x == vertex, vertices):
                vertices_added_to_component.add(k)
        component_id += 1
    return component_for_vertex
