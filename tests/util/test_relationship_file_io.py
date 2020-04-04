import os
import tempfile
import redblackgraph as rb

from scipy.sparse import coo_matrix
from numpy.testing import assert_equal

from redblackgraph.sparse.csgraph import transitive_closure
from redblackgraph.reference.ordering import avos_canonical_ordering


def test_rel_file():
    test_persons_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.vertices.csv")
    test_relationships_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.edges.csv")
    reader = rb.RelationshipFileReader(test_persons_file, test_relationships_file, 4, ["Bio"])
    graph = reader.read()
    vertex_key = reader.get_vertex_key()

    # expected graph
    expected = rb.rb_matrix(coo_matrix((
            [-1, 3, 2, 1,-1, 3, 2, 1, 2, 3,-1, 2, 3,-1, 1, 1,-1, 2, 3,-1, 2, 3,-1, 2, 3,-1, 1, 1, 1],
        (
            [ 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 9, 9, 9,10,10,10,11,12,13,14],
            [ 0, 1, 4, 1, 2, 3, 9, 3, 4, 7, 4, 5, 6, 5, 6, 7, 8, 9,14, 9,10,13,10,11,12,11,12,13,14]
        )
    )))
    for i, j in zip(*graph.nonzero()):
        assert_equal(graph[i, j], expected[i, j])
    for i, j in zip(*expected.nonzero()):
        assert_equal(graph[i, j], expected[i, j])

    # expected vertex key
    expected = {
        8: ('ABCD-1AB', 'R, H'),
        2: ('ABCD-2AB', 'R, D'),
        3: ('ABCD-3AB', 'S, R'),
        0: ('ABCD-4AB', 'S, D'),
        9: ('ABCD-5AB', 'R, E'),
        14: ('ABCD-6AB', 'T, M'),
        13: ('ABCD-7AB', 'W, A'),
        10: ('ABCD-8AB', 'R, M'),
        7: ('ABCD-9AB', 'W, I'),
        4: ('ABCD-AAB', 'S, J'),
        1: ('ABCD-BAB', 'K, E'),
        12: ('ABCD-CAB', 'K, M'),
        11: ('ABCD-DAB', 'R, G'),
        5: ('ABCD-EAB', 'S, S'),
        6: ('ABCD-FAB', 'C, E'),
    }
    for key, value in vertex_key.items():
        assert key in expected
        assert value[0] == expected[key][0]
        assert value[1] == expected[key][1]
    assert reader.get_person_id('ABCD-1AB') == 0

    writer = rb.RedBlackGraphWriter(reader)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'test_file.xlsx')
        writer.write(graph, output_file=tmpfile)
        assert os.path.isfile(tmpfile)

        R_star = transitive_closure(graph).W
        R_cannonical = avos_canonical_ordering(R_star)

        tmpfile_cannonical = os.path.join(tmpdir, 'test_file_cannonical.xlsx')
        writer.write(R_cannonical.A, output_file=tmpfile_cannonical, key_permutation=R_cannonical.label_permutation)
        # TODO: really should figure out a way to test the xlsx headers were written correctly...
        assert os.path.isfile(tmpfile_cannonical)
