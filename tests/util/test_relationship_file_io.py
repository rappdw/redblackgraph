import os
import tempfile
import redblackgraph as rb
from redblackgraph.reference.triangularization import canonical_sort

def test_rel_file():
    test_persons_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.vertices.csv")
    test_relationships_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.edges.csv")
    reader = rb.RelationshipFileReader(test_persons_file, test_relationships_file)
    graph: rb.array = reader()
    assert len(graph) == len(graph[0])
    assert len(graph) == 15
    vertex_key = reader.get_vertex_key()
    assert len(vertex_key.keys()) == len(graph)
    assert reader.get_person_id('ABCD-1AB') == 0

    writer = rb.RedBlackGraphWriter(reader)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'test_file.xlsx')
        writer(graph, output_file=tmpfile)
        assert os.path.isfile(tmpfile)

        R_star = graph.transitive_closure().W
        R_cannonical = canonical_sort(R_star)

        tmpfile_cannonical = os.path.join(tmpdir, 'test_file_cannonical.xlsx')
        writer(R_cannonical.A, output_file=tmpfile_cannonical, key_permutation=R_cannonical.label_permutation)
        # TODO: really should figure out a way to test the xlsx headers were written correctly...
        assert os.path.isfile(tmpfile_cannonical)
