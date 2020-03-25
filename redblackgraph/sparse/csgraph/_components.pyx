import numpy as np
cimport numpy as np

import itertools as it

from dataclasses import dataclass
from redblackgraph.core.redblack import array as rb_array
from typing import List
from collections import defaultdict
from typing import Dict, Optional, Sequence

include 'parameters.pxi'
include '_rbg_math.pxi'

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
    # Allocate a set that contains the vertices visited
    # Iterate over each row not in the visited vertices:
    #   This is a new component, so increment the component id and assign the vertex of the current row to that id
    #   Allocate a set that holds vertices that will be added to this component
    #   In a given row, iterate over the columns not in the visited vertices (since this is a new component,
    #   no columns will be in the visited vertices):
    #     Any non-zero columns in that row will be assigned to the row component and added to the set of added vertices
    #

    n = len(A)
    if q is None:
        q = defaultdict(lambda: 0)
    component_for_vertex = [0] * n
    vertices = range(n)
    visited_vertices = set()
    component_id = 0
    for i in it.filterfalse(lambda x: x in visited_vertices, vertices):
        vertices_added_to_component = set()
        vertex_count = 0
        vertices_added_to_component.add(i)
        while vertices_added_to_component:
            vertex = vertices_added_to_component.pop()
            vertex_count += 1
            visited_vertices.add(vertex)
            component_for_vertex[vertex] = component_id
            for j in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[vertex][x] == 0 or x == vertex, vertices):
                vertices_added_to_component.add(j)
                for k in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[x][j] == 0 or x == j, vertices):
                    vertices_added_to_component.add(k)
            # now we need to iterate the vertex's column
            for k in it.filterfalse(lambda x: x in visited_vertices or x in vertices_added_to_component or A[x][vertex] == 0 or x == vertex, vertices):
                vertices_added_to_component.add(k)
        q[component_id] = vertex_count
        component_id += 1
    return component_for_vertex

class Components:

    def __init__(self, A, ids, rel_count, max_rel, size_map):
        self.A = A
        self.ids = ids
        self.rel_count = rel_count
        self.max_rel = max_rel
        self.size_map = size_map

    def get_ordering(self) -> List[int]:
        # This is the default sort ordering used by Traingularization
        # it sorts by:
        #   * size of component, descending
        #   * component id, ascending
        #   * relationship count, descending
        #   * max ancestor: ascending
        #   * color: descending
        #   * vertex_id: ascending
        basis = [i for i in range(len(self.ids))]
        # sort descending on size of component and "ancestor count", ascending on all other elements
        basis.sort(key=lambda x: (-self.size_map[self.ids[x]], self.ids[x], -self.rel_count[x], self.max_rel[x], -self.A[x][x], x))
        return basis

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
    # 3 seconds exeuction time (3633 vertices)

    q = defaultdict(lambda: 0)
    component_for_vertex = find_components(A, q)

    n = len(A)
    max_rel_for_vertex = [0] * n
    ancester_count_for_vertex = [0] * n
    vertices = range(n)
    for i in vertices:
        for j in it.filterfalse(lambda x: (A[i][x] == 0 and A[x][i] == 0) or x == i, vertices):
            max_rel_for_vertex[i] = max(max_rel_for_vertex[i], A[j][i])
            ancester_count_for_vertex[i] += MSB(A[i][j])

    return Components(A, component_for_vertex, ancester_count_for_vertex, max_rel_for_vertex, {k:v for k,v in q.items() if v != 0})
