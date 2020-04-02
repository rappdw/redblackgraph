import numpy as np
import redblackgraph as rb

import redblackgraph.reference as ref
import redblackgraph.core as core
import redblackgraph.sparse as sparse

import pytest
from numpy.testing import assert_equal

def create_ref_array(A, dtype=np.int32, shape=None):
    if shape:
        return rb.array(A, dtype=dtype).reshape(shape).tolist()
    return A

def create_core_array(A, dtype=np.int32, shape=None):
    if shape:
        return rb.array(A, dtype=dtype).reshape(shape)
    return rb.array(A, dtype=dtype)

@pytest.mark.parametrize("dtype", [
    np.int32
])
@pytest.mark.parametrize("vertex_relational_composition,edge_relational_composition,create_array", [
    (ref.vertex_relational_composition, ref.edge_relational_composition, create_ref_array),
    (core.vertex_relational_composition, core.edge_relational_composition, create_core_array),
    (sparse.vertex_relational_composition, sparse.edge_relational_composition, create_core_array),
])
def test_vertex_relational_composition(vertex_relational_composition, edge_relational_composition, create_array, dtype):
    # use the canonical ordering of A+ from the example in our notebook,
    # add in a sibling to the last vertex in the graph
    A = create_array([[-1, 2, 3, 4, 0],
                      [ 0,-1, 0, 2, 0],
                      [ 0, 0, 1, 0, 0],
                      [ 0, 0, 0,-1, 0],
                      [ 2, 4, 5, 8, 1]], dtype=dtype)

    u = create_array([2, 0, 0, 0, 0], dtype=dtype, shape=(1, 5))
    v = create_array([0, 0, 0, 0, 0], dtype=dtype, shape=(5, 1))
    A_lambda = vertex_relational_composition(u, A, v, -1)
    expected_1 = create_array([[-1, 2, 3, 4, 0, 0],
                               [ 0,-1, 0, 2, 0, 0],
                               [ 0, 0, 1, 0, 0, 0],
                               [ 0, 0, 0,-1, 0, 0],
                               [ 2, 4, 5, 8, 1, 0],
                               [ 2, 4, 5, 8, 0,-1]], dtype=dtype)
    assert_equal(A_lambda, expected_1)

    # Using the A_lambda that was generated... Add in the "great-grandmother" to the siblings represented by vertex 4 and 5
    u = create_array([0, 0, 0, 0, 0, 0], dtype=dtype, shape=(1, 6))
    v = create_array([0, 3, 0, 0, 0, 0], dtype=dtype, shape=(6, 1))
    A_lambda = vertex_relational_composition(u, A_lambda, v, 1)
    expected_2 = create_array([[-1, 2, 3, 4, 0, 0, 5],
                               [ 0,-1, 0, 2, 0, 0, 3],
                               [ 0, 0, 1, 0, 0, 0, 0],
                               [ 0, 0, 0,-1, 0, 0, 0],
                               [ 2, 4, 5, 8, 1, 0, 9],
                               [ 2, 4, 5, 8, 0,-1, 9],
                               [ 0, 0, 0, 0, 0, 0, 1]], dtype=dtype)
    assert_equal(A_lambda, expected_2)

@pytest.mark.parametrize("vertex_relational_composition,edge_relational_composition,create_array", [
    (ref.vertex_relational_composition, ref.edge_relational_composition, create_ref_array),
    (core.vertex_relational_composition, core.edge_relational_composition, create_core_array),
    (sparse.vertex_relational_composition, sparse.edge_relational_composition, create_core_array),
])
def test_my_use_case_vertex(vertex_relational_composition,edge_relational_composition,create_array):
    #        D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
    A = create_array([
          [ -1,  2,  3,  0,  0,  4,  5,  7,  0,  0,  8,  9,  0,  0],  # D
          [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  4,  5,  0,  0],  # E
          [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],  # R
          [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0,  3, -1,  4,  5,  0,  0,  0,  8,  9,  0,  0],  # H
          [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0],  # Mi
          [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  0,  0],  # Do
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # Ev
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],  # G
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # Ma
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],  # S
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]   # Em
    ])
    #              D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
    expected = create_array([
                [ -1,  2,  3,  0,  0,  4,  5,  7,  0,  0,  8,  9, 12, 13,  6],  # D
                [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  4,  5,  0,  0,  0],  # E
                [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  4,  5,  2],  # R
                [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
                [  0,  2,  0,  3, -1,  4,  5,  0,  0,  0,  8,  9,  0,  0,  0],  # H
                [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
                [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
                [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
                [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  4,  5,  2],  # Do
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # Em
                [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1],  # J
    ])

    #                  D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em
    u = create_array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3], shape=(1, 14))
    v = create_array([ 0,  0,  2,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0], shape=(14, 1))


    A_lambda = vertex_relational_composition(u, A, v, -1)
    assert_equal(A_lambda, expected)

@pytest.mark.parametrize("vertex_relational_composition,edge_relational_composition,create_array", [
    (ref.vertex_relational_composition, ref.edge_relational_composition, create_ref_array),
    (core.vertex_relational_composition, core.edge_relational_composition, create_core_array),
    (sparse.vertex_relational_composition, sparse.edge_relational_composition, create_core_array),
])
def test_edge_relational_composition_simple(vertex_relational_composition,edge_relational_composition,create_array):
    R = create_array([[-1, 0, 3, 0, 0],
                      [ 0,-1, 0, 2, 0],
                      [ 0, 0, 1, 0, 0],
                      [ 0, 0, 0,-1, 0],
                      [ 2, 0, 5, 0, 1]])
    R_lambda = edge_relational_composition(R, 0, 1, 2)
    R_expected = create_array([[-1, 2, 3, 4, 0],
                               [ 0,-1, 0, 2, 0],
                               [ 0, 0, 1, 0, 0],
                               [ 0, 0, 0,-1, 0],
                               [ 2, 4, 5, 8, 1]])
    assert_equal(R_lambda, R_expected)

@pytest.mark.parametrize("vertex_relational_composition,edge_relational_composition,create_array", [
    (ref.vertex_relational_composition, ref.edge_relational_composition, create_ref_array),
    (core.vertex_relational_composition, core.edge_relational_composition, create_core_array),
    (sparse.vertex_relational_composition, sparse.edge_relational_composition, create_core_array),
])
def test_my_use_case_edge(vertex_relational_composition,edge_relational_composition,create_array):
    #        D   E   R   M   H  Mi   A   I  Do  Ev   G  Ma   S  Em   J
    A = create_array([
          [ -1,  2,  3,  0,  0,  4,  5,  7,  0,  0,  8,  9,  0,  0,  0],  # D
          [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  4,  5,  0,  0,  0],  # E
          [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],  # R
          [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0,  3, -1,  4,  5,  0,  0,  0,  8,  9,  0,  0,  0],  # H
          [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
          [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  4,  5,  2],  # Do
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # Em
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]   # J
    ])
    expected = create_array([
          [ -1,  2,  3,  0,  0,  4,  5,  7,  0,  0,  8,  9, 12, 13,  6],  # D
          [  0, -1,  0,  0,  0,  2,  3,  0,  0,  0,  4,  5,  0,  0,  0],  # E
          [  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  0,  4,  5,  2],  # R
          [  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # M
          [  0,  2,  0,  3, -1,  4,  5,  0,  0,  0,  8,  9,  0,  0,  0],  # H
          [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  3,  0,  0,  0],  # Mi
          [  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # A
          [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # I
          [  0,  0,  0,  0,  0,  0,  0,  0, -1,  3,  0,  0,  4,  5,  2],  # Do
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # Ev
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # G
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # Ma
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # S
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # Em
          [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  3, -1]   # J
    ])

    # Missing edge is R -> J, 2
    A_lambda = edge_relational_composition(A, 2, 14, 2)
    assert_equal(A_lambda, expected)