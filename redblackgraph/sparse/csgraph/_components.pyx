import numpy as np
cimport numpy as np
cimport cython

from redblackgraph.core.redblack import array as rb_array
from typing import Dict, Optional, Sequence

include 'parameters.pxi'
include '_rbg_math.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
def find_components(A: rb_array, q:Optional[Dict[int, int]] = None) -> Sequence[int]:
    """
    Given an input adjacency matrix compute the connected components
    :param A: input adjacency matrix (this implementation assumes that it is transitively closed)
    :param q: if set, should be defaultdict(lambda: 0)
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
    # Allocate an array that will represent the vertices that have been visited
    # Iterate over each vertex that hasn't been visited:
    #   This is a new component, so increment the component id
    #   Allocate a set to hold ids to be added to this component
    #   Add the current vertex to this set
    #   while the set is not empty
    #     pull a vextex from the set
    #     add it to the current component
    #     For each non-zero cell in the vertex's row and column add those vertices to the set for this component

    cdef unsigned int n = len(A)
    vertices = range(n)
    component_for_vertex_np = np.zeros((n), dtype=np.uint32)
    cdef unsigned int[ : ] component_for_vertex = component_for_vertex_np
    cdef unsigned char[ : ] visited_vertices = np.zeros((n), dtype=np.uint8)
    cdef unsigned int component_id = 0
    cdef unsigned int vertex_count
    cdef DTYPE_t[:, :] Am = A
    for i in vertices: # it.filterfalse(lambda x: visited_vertices[x], vertices):
        if visited_vertices[i]:
            continue
        vertices_added_to_component = set()
        vertex_count = 0
        vertices_added_to_component.add(i)
        while vertices_added_to_component:
            vertex = vertices_added_to_component.pop()
            vertex_count += 1
            visited_vertices[vertex] = True
            component_for_vertex[vertex] = component_id
            for j in vertices:
                if not ((Am[vertex][j] == 0 and Am[j][vertex] == 0) or visited_vertices[j] or j in vertices_added_to_component):
                    vertices_added_to_component.add(j)
        if q is not None:
            q[component_id] = vertex_count
        component_id += 1
    return component_for_vertex_np
