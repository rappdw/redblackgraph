import os
import tempfile
import redblackgraph as rb
from redblackgraph.sparse.csgraph import transitive_closure
from redblackgraph.reference.triangularization import canonical_sort

def test_rel_file():
    test_persons_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.vertices.csv")
    test_relationships_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.edges.csv")
    reader = rb.RelationshipFileReader(test_persons_file, test_relationships_file, 4, ["Bio"])
    graph = reader.read()
    assert graph.shape[0] == graph.shape[1]
    assert graph.shape[0] == 15
    vertex_key = reader.get_vertex_key()
    assert len(vertex_key.keys()) == graph.shape[0]
    assert reader.get_person_id('ABCD-1AB') == 0

    writer = rb.RedBlackGraphWriter(reader)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'test_file.xlsx')
        writer.write(graph, output_file=tmpfile)
        assert os.path.isfile(tmpfile)

        R_star = transitive_closure(graph).W
        R_cannonical = canonical_sort(R_star)

        tmpfile_cannonical = os.path.join(tmpdir, 'test_file_cannonical.xlsx')
        writer.write(R_cannonical.A, output_file=tmpfile_cannonical, key_permutation=R_cannonical.label_permutation)
        # TODO: really should figure out a way to test the xlsx headers were written correctly...
        assert os.path.isfile(tmpfile_cannonical)
