import numpy as np
import redblackgraph.reference as ref
import redblackgraph.sparse.csgraph as sparse
import pytest

from numpy.testing import assert_equal

from collections import defaultdict

def find_components(A, q):
    return sparse.find_components(np.array(A, dtype=np.int32), q)

def get_permutation(A, q, components):
    return sparse._get_permutation(np.array(A, dtype=np.int32), q, components)

@pytest.mark.parametrize(
    "transitive_closure,find_components,get_permutation",
    [
        (ref.transitive_closure, ref.find_components, ref.ordering._get_permutation),
        (sparse.transitive_closure_floyd_warshall, find_components, get_permutation),
        (sparse.transitive_closure_dijkstra, find_components, get_permutation),
    ]
)
def test_ordering(transitive_closure, find_components, get_permutation):
    vertex_key = {0: (10, 'DWR'),
                   1: (11, 'LEJ'),
                   2: (0, 'ISR'),
                   3: (19, 'BAV'),
                   4: (14, 'RS'),
                   5: (2, 'MJR'),
                   6: (12, 'ER'),
                   7: (1, 'NAR'),
                   8: (8, 'HDR'),
                   9: (18, 'ILW'),
                   10: (20, 'AAV'),
                   11: (4, 'WER'),
                   12: (9, 'MR'),
                   13: (7, 'GHR'),
                   14: (21, 'IG'),
                   15: (15, 'MJR I'),
                   16: (13, 'MT'),
                   17: (3, 'MHR'),
                   18: (5, 'DR'),
                   19: (6, 'UCR'),
                   20: (16, 'AW'),
                   21: (17, 'JWS')}
    A = [
      [-1, 0, 0, 0, 3, 0, 2, 0, 0, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 6],
      [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [ 2, 3,-1, 0, 5, 0, 4, 0, 0,11, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9,10],
      [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
      [ 2, 3, 0, 0, 5,-1, 4, 0, 0,11, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9,10],
      [ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0],
      [ 2, 3, 0, 0, 5, 0, 4, 1, 0,11, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9,10],
      [ 0, 0, 0, 0, 0, 0, 2, 0,-1, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 5, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,-1, 0, 0, 0, 4, 3, 0, 0, 0, 5, 0],
      [ 0, 0, 0, 0, 3, 0, 2, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 5, 6],
      [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,-1, 0, 4, 3, 0, 0, 0, 5, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3,-1, 0, 0, 5, 0],
      [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0,-1, 0, 5, 0],
      [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 1, 5, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
    ]
    q = dict() # dictionary keyed by component id, value is count of vertices in componenet
    components = find_components(A, q)
    #vertices =                    [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21]
    assert_equal(components,       [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # which component does a vertex belong to
    assert q[0] == 19
    assert q[1] == 3

    permutation = get_permutation(A, q, components)

    # the following vertex groups have components that are indistinguishable aside from vertex id and color (in some cases)
    # [2, 5, 7]
    # [8, 11, 13, 17, 18, 19]
    # [1, 16]
    # so, whatever the ordering, this should be adjacent
    group0 = {2, 5, 7}
    group1 = {8, 11, 13, 17, 18, 19}
    group2 = {1, 16}
    indistinguishable = [
        [False, group0, -1],
        [False, group1, -1],
        [False, group2, -1],
    ]
    for idx, element in enumerate(permutation):
        for i, [scanned, group, _] in enumerate(indistinguishable):
            if not scanned and element in group:
                for j in range(len(group)):
                    assert permutation[j + idx] in group
                indistinguishable[i][0] = True
                indistinguishable[i][-1] = idx
    # this next set of assertions is a bit implementation specific rather than validating the invariants,
    # but keep it in for now
    assert indistinguishable[0][-1] < indistinguishable[1][-1]
    assert indistinguishable[1][-1] < indistinguishable[2][-1]

    # whatever the specific ordering is the following topological ordering needs to be observed
    # pos[2] < pos[0] < pos[4] < pos[21]
    # pos[8] < pos[6] < pos[20]

    vertices_to_verify = {0, 2, 4, 6, 8, 20, 21}
    topological_order = dict()
    for idx, element in enumerate(permutation):
        if element in vertices_to_verify:
            topological_order[element] = idx
    assert topological_order[2] < topological_order[0]
    assert topological_order[0] < topological_order[4]
    assert topological_order[4] < topological_order[21]

    assert topological_order[8] < topological_order[6]
    assert topological_order[6] < topological_order[20]
