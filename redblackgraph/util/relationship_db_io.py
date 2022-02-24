import sqlite3 as sl
import time
from typing import Dict, Tuple

import numpy as np
import redblackgraph as rb

from scipy.sparse import coo_matrix
from .relationship_file_io import VertexInfo
from redblackgraph.sparse.csgraph import avos_canonical_ordering
from redblackgraph.types import Ordering

unordered_vertices = "SELECT ROWID, color FROM VERTEX;"
unordered_edge_count = """
    select count(*)
    from VERTEX
    join (
        select VERTEX.ROWID as src_id, destination from VERTEX
        JOIN EDGE on source = VERTEX.ID
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent')
    ) ON VERTEX.id = destination
    order by src_id;
"""
unordered_edges = """
    select src_id, VERTEX.ROWID as dst_id
    from VERTEX
    join (
        select VERTEX.ROWID as src_id, destination from VERTEX
        JOIN EDGE on source = VERTEX.ID
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent')
    ) ON VERTEX.id = destination
    order by src_id;
"""
ordered_vertices = """
    SELECT position, color from VERTEX 
    join ORDERING on ORDERING.id = VERTEX.ROWID 
    ORDER BY position DESC;
"""
ordered_edges = """
    select src_position, position as dst_position
    from VERTEX
    join (
        select position as src_position, destination from VERTEX
        JOIN EDGE on source = VERTEX.ID
        JOIN ORDERING on ORDERING.id = VERTEX.ROWID
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent')
    ) ON VERTEX.id = destination
    JOIN ORDERING on ORDERING.id = VERTEX.ROWID
    order by src_position DESC, dst_position DESC;
"""
vertex_key_query = """
SELECT position, VERTEX.id, given_name, surname from VERTEX
join ORDERING on ORDERING.id = VERTEX.ROWID
ORDER BY position;
"""
create_ordering_table = """
CREATE TABLE IF NOT EXISTS ORDERING (
        id INTEGER NOT NULL PRIMARY KEY,
        position INTEGER
        );
"""
create_ordering_index = """
CREATE INDEX IF NOT EXISTS ORDER_INDEX ON ORDERING(position)
"""



def _read_graph_dense(conn, nv, vertex_query, edge_query):
    graph = rb.matrix(np.zeros((nv, nv), dtype=np.int64))
    vertices = conn.execute(vertex_query)
    for vertex in vertices:
        # get the vertex number (first column) and color (second column)
        i = vertex[0] - 1
        graph[i, i] = vertex[1]

    # ne = conn.execute(unordered_edge_count).fetchone()[0]
    edges = conn.execute(edge_query)
    for edge in edges:
        # get the source (first column) and destination (second column)
        i = edge[0] - 1
        j = edge[1] - 1
        # set the relationship to 2 (father), 3 (mother) based on color of vertex
        if graph[j, j] == 1:
            graph[i, j] = 3
        else:
            graph[i, j] = 2
    return graph


def _read_graph_sparse(conn, nv, vertex_query, edge_query):
    ne = conn.execute(unordered_edge_count).fetchone()[0]
    val = np.zeros(ne + nv, dtype=np.int64)
    row = np.zeros(ne + nv, dtype=np.int64)
    col = np.zeros(ne + nv, dtype=np.int64)
    genders = np.zeros(nv, dtype=np.int64)

    vertices = conn.execute(vertex_query)
    edges = conn.execute(edge_query)

    idx = ne + nv - 1
    read_vertex = True
    read_edge = True
    i = color = src = dst = None

    while True:
        if read_vertex:
            try:
                vertex = next(vertices)
                i = vertex[0] - 1
                color = vertex[1]
                genders[i] = color
                read_vertex = False
            except StopIteration:
                pass
        if read_edge:
            try:
                edge = next(edges)
                src = edge[0] - 1
                dst = edge[1] - 1
                read_edge = False
            except StopIteration:
                # we've consumed the last edge, set the src less than zero so that we can process
                # the last vertex
                src = -1
        if src < i:
            val[idx] = color
            row[idx] = i
            col[idx] = i
            read_vertex = True
        else:
            val[idx] = 3 if genders[dst] == 1 else 2
            row[idx] = src
            col[idx] = dst
            read_edge = True
        idx -= 1
        if idx < 0:
            break
    graph = rb.rb_matrix(coo_matrix((val, (row, col))))
    return graph


def _read_graph(conn, nv, vertex_query, edge_query, sparse_size_threshold):
    if nv > sparse_size_threshold:
        return _read_graph_sparse(conn, nv, vertex_query, edge_query)
    else:
        return _read_graph_dense(conn, nv, vertex_query, edge_query)


class RelationshipDbReader(VertexInfo):

    # TODO: support hops

    def __init__(self, db_file, hops):
        self.db_file = db_file
        self.hops = hops

    def read(self, sparse_size_threshold=1000):
        conn = sl.connect(self.db_file)
        nv = conn.execute("SELECT COUNT(*) FROM VERTEX").fetchone()[0]
        ordering_table_q = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='ORDERING'")
        table_exists = ordering_table_q.fetchone()[0] == 1
        ordering_count = 0
        if table_exists:
            ordering_count = conn.execute("SELECT COUNT(*) FROM ORDERING").fetchone()[0]
        if not table_exists or ordering_count != nv:
            # reading the graph using unordered vertices and edges and then, run transitive closure
            # and then canonical ordering
            g = _read_graph(conn, nv, unordered_vertices, unordered_edges, sparse_size_threshold)
            ordering = avos_canonical_ordering(g.transitive_closure().W)
            self.save_ordering(conn, ordering)
        return _read_graph(conn, nv, ordered_vertices, ordered_edges, sparse_size_threshold)

    def save_ordering(self, conn, ordering: Ordering):
        conn.execute(create_ordering_table)
        conn.execute(create_ordering_index)
        conn.execute("DELETE FROM ORDERING")
        for idx, id in enumerate(ordering.label_permutation):
            conn.execute(f"INSERT INTO ORDERING (id, position) values({id + 1}, {idx + 1})")
        conn.commit()


    def get_vertex_key(self) -> Dict[int, Tuple[str, str]]:
        """
        Get the "vertex key", a dictionary keyed by the vertex id (int) with values
        that are tuples of (external id, string designation)
        """
        conn = sl.connect(self.db_file)
        vertices = conn.execute(vertex_key_query)
        vertex_key = dict()
        for vertex in vertices:
            vertex_id = vertex[0] - 1
            external_id = vertex[1]
            vertex_name = f"'{vertex[3]}', '{vertex[2]}'"
            vertex_key[vertex_id] = (external_id, vertex_name)
        return vertex_key


if __name__ == "__main__":
    rdr = rb.RelationshipDbReader("/Users/drapp/data/rbg/rappdw.db", 4)
    g = rdr.read(1000)
    start = time.time()
    closure = g.transitive_closure()
    duration = time.time() - start
    print(f"duration: {duration}")
