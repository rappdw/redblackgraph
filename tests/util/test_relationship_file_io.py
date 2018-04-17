import os
import tempfile
import redblackgraph as rb

def test_rel_file():
    test_file = os.path.join(os.path.dirname(__file__), "resources/sample-tree.csv")
    reader = rb.RelationshipFileReader(test_file)
    graph: rb.array = reader()
    assert len(graph) == len(graph[0])
    assert len(graph) == 15
    vertex_key = reader.get_vertex_key()
    assert len(vertex_key.keys()) == len(graph)
    assert reader.get_person_id(('D', 'R', '1963')) == 0

    writer = rb.RedBlackGraphWriter(vertex_key)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, 'test_file.xlsx')
        writer(graph, output_file=tmpfile)
        assert os.path.isfile(tmpfile)

        R_star = graph.transitive_closure()[0]
        P = R_star.get_triangularization_permutation_matrices()
        R_cannonical = R_star.triangularize(P)

        tmpfile_cannonical = os.path.join(tmpdir, 'test_file_cannonical.xlsx')
        writer(R_cannonical, output_file=tmpfile_cannonical, key_permutation=P[2])
        # TODO: really should figure out a way to test the xlsx headers were written correctly...
        assert os.path.isfile(tmpfile_cannonical)