"""Read a relationship database (sqlite) and drive a graph builder.

Vendored from rappdw/fs-crawler (``fscrawler/util/db_reader.py``) at commit
fae46e76e3849924d6d1fb96f343f9a00a21ca98. fs-crawler is deprecated as a
dependency; only the graph-consumption side it provided lives on here. The
original FamilySearch crawler that produces the relationship databases remains
external. fs-crawler is licensed GPL-3.0-or-later, compatible with this
project's AGPLv3+.
"""

import sqlite3 as sl
import logging
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple
from .abstract_graph import AbstractGraphBuilder, VertexInfo

unordered_vertices = "SELECT ROWID, color FROM VERTEX;"
unordered_edge_count = """
    select count(*)
    from VERTEX
    join (
        select VERTEX.ROWID as src_id, destination from VERTEX
        JOIN EDGE on source = VERTEX.ID
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
    ) ON VERTEX.id = destination
    order by src_id;
"""
unordered_edges = """
    select src_id, VERTEX.ROWID as dst_id
    from VERTEX
    join (
        select VERTEX.ROWID as src_id, destination from VERTEX
        JOIN EDGE on source = VERTEX.ID
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
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
        WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
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


class RelationshipDbReader(VertexInfo):

    def __init__(self, db_file, hops, graph_builder: AbstractGraphBuilder):
        self.db_file = db_file
        self.hops = hops
        self.graph_builder = graph_builder
        self.logger = logging.getLogger(self.__class__.__name__)

    def _check_iteration_data(self, conn) -> bool:
        """
        Check if the database has iteration data available for hop filtering.
        Returns True if iteration field is populated for vertices.
        """
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM VERTEX WHERE iteration IS NOT NULL"
            ).fetchone()[0]
            return result > 0
        except sl.OperationalError:
            # iteration column doesn't exist
            return False

    def _get_filtered_queries(self, hops: int) -> dict:
        """
        Generate SQL queries filtered by iteration (hop count).
        Only includes vertices with iteration < hops.
        """
        iteration_filter = f"iteration < {hops}"

        filtered_unordered_vertices = f"SELECT ROWID, color FROM VERTEX WHERE {iteration_filter};"

        filtered_unordered_edge_count = f"""
            select count(*)
            from VERTEX
            join (
                select VERTEX.ROWID as src_id, destination from VERTEX
                JOIN EDGE on source = VERTEX.ID
                WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
                  AND VERTEX.{iteration_filter}
            ) ON VERTEX.id = destination
            WHERE VERTEX.{iteration_filter}
            order by src_id;
        """

        filtered_unordered_edges = f"""
            select src_id, VERTEX.ROWID as dst_id
            from VERTEX
            join (
                select VERTEX.ROWID as src_id, destination from VERTEX
                JOIN EDGE on source = VERTEX.ID
                WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
                  AND VERTEX.{iteration_filter}
            ) ON VERTEX.id = destination
            WHERE VERTEX.{iteration_filter}
            order by src_id;
        """

        filtered_ordered_vertices = f"""
            SELECT position, color from VERTEX
            join ORDERING on ORDERING.id = VERTEX.ROWID
            WHERE VERTEX.{iteration_filter}
            ORDER BY position DESC;
        """

        filtered_ordered_edges = f"""
            select src_position, position as dst_position
            from VERTEX
            join (
                select position as src_position, destination from VERTEX
                JOIN EDGE on source = VERTEX.ID
                JOIN ORDERING on ORDERING.id = VERTEX.ROWID
                WHERE EDGE.type in ('AssumedBiological', 'UnspecifiedParentType', 'BiologicalParent', 'UntypedParent')
                  AND VERTEX.{iteration_filter}
            ) ON VERTEX.id = destination
            JOIN ORDERING on ORDERING.id = VERTEX.ROWID
            WHERE VERTEX.{iteration_filter}
            order by src_position DESC, dst_position DESC;
        """

        filtered_vertex_key = f"""
            SELECT position, VERTEX.id, given_name, surname from VERTEX
            join ORDERING on ORDERING.id = VERTEX.ROWID
            WHERE VERTEX.{iteration_filter}
            ORDER BY position;
        """

        return {
            'unordered_vertices': filtered_unordered_vertices,
            'unordered_edge_count': filtered_unordered_edge_count,
            'unordered_edges': filtered_unordered_edges,
            'ordered_vertices': filtered_ordered_vertices,
            'ordered_edges': filtered_ordered_edges,
            'vertex_key': filtered_vertex_key,
        }

    def compute_ordering(self) -> None:
        """Compute and save the canonical ordering for the graph.

        This method should be called before read() if the ordering table doesn't exist
        or needs to be updated. It reads the unordered graph, computes the canonical
        ordering, and saves it to the database.
        """
        self.logger.info(f"Computing canonical ordering for: {self.db_file}")
        conn = sl.connect(self.db_file)

        # Get vertex and edge counts for full graph
        nv = conn.execute("SELECT COUNT(*) FROM VERTEX").fetchone()[0]
        ne = conn.execute(unordered_edge_count).fetchone()[0]

        # Check if ordering already exists and is up to date
        ordering_table_q = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='ORDERING'")
        table_exists = ordering_table_q.fetchone()[0] == 1
        ordering_count = 0
        if table_exists:
            ordering_count = conn.execute("SELECT COUNT(*) FROM ORDERING").fetchone()[0]

        if table_exists and ordering_count == nv:
            self.logger.info("Canonical ordering already exists and is up to date")
            return

        # Compute the ordering
        start_time = time.time()
        self.logger.info(f"Graph size: {nv:,} vertices and {ne:,} edges")
        self.graph_builder.init_builder(nv, ne)
        self.logger.info("Reading unordered graph...")
        self._read_graph(conn, unordered_vertices, unordered_edges)
        self.logger.info("Computing ordering...")
        ordering = self.graph_builder.get_ordering()
        self.logger.info("Saving ordering to database...")
        self.save_ordering(conn, ordering)
        self.logger.info(f"Canonical ordering computed and saved in {time.time() - start_time:.2f} seconds")

    def _get_cache_path(self) -> Path:
        """Get the cache file path based on the database file and hop filtering."""
        db_path = Path(self.db_file)
        if self.hops is not None:
            cache_name = f"{db_path.stem}_hops{self.hops}.cache"
        else:
            cache_name = f"{db_path.stem}.cache"
        return db_path.parent / cache_name

    def _is_cache_valid(self) -> bool:
        """Check if the cache file exists and is newer than the database.

        Delegates to the graph_builder's is_cache_valid method to avoid
        implementation details leaking into this base class.
        """
        cache_path = self._get_cache_path()
        db_path = Path(self.db_file)

        # Delegate to graph builder for format-specific validation
        return self.graph_builder.is_cache_valid(cache_path, db_path)

    def save_cache(self, graph) -> None:
        """Save the graph to a cache file for fast loading.

        Args:
            graph: The built graph object to cache
        """
        cache_path = self._get_cache_path()
        self.logger.info(f"Saving graph cache to: {cache_path}")
        start_time = time.time()

        # Delegate to graph builder with metadata
        metadata = {'hops': self.hops}
        self.graph_builder.save_cache(graph, cache_path, metadata)

        self.logger.info(f"Graph cache saved in {time.time() - start_time:.2f} seconds")

    def load_cache(self):
        """Load the graph from cache file.

        Returns:
            The cached graph object

        Raises:
            ValueError: If the cached hop count doesn't match the current hop count
        """
        cache_path = self._get_cache_path()
        self.logger.info(f"Loading graph from cache: {cache_path}")
        start_time = time.time()

        # Delegate to graph builder with expected metadata
        expected_metadata = {'hops': self.hops}
        graph = self.graph_builder.load_cache(cache_path, expected_metadata)

        self.logger.info(f"Graph loaded from cache in {time.time() - start_time:.2f} seconds")
        return graph

    def read(self, use_cache: bool = False):
        """Read the graph from the database using the canonical ordering.

        Args:
            use_cache: If True, attempt to load from cache. If cache is invalid or doesn't exist,
                      read from database and save to cache.

        Note: This method assumes the canonical ordering has already been computed.
        Call compute_ordering() first if the ordering table doesn't exist.
        """
        # Try to use cache if requested
        if use_cache:
            if self._is_cache_valid():
                try:
                    return self.load_cache()
                except (ValueError, Exception) as e:
                    self.logger.warning(f"Failed to load cache: {e}. Reading from database...")
            else:
                self.logger.info("Cache not found or outdated, reading from database...")
        self.logger.info(f"Reading relationship database: {self.db_file}")
        conn = sl.connect(self.db_file)

        # Check if hop filtering should be applied
        use_hop_filtering = False
        if self.hops is not None:
            if self._check_iteration_data(conn):
                use_hop_filtering = True
                self.logger.info(f"Applying hop filtering: iteration < {self.hops}")
            else:
                self.logger.warning(f"Hop filtering requested (hops={self.hops}) but iteration data not available. Reading entire graph.")

        # Get appropriate queries based on hop filtering
        if use_hop_filtering:
            filtered_queries = self._get_filtered_queries(self.hops)
            nv = conn.execute(f"SELECT COUNT(*) FROM VERTEX WHERE iteration < {self.hops}").fetchone()[0]
            ne = conn.execute(filtered_queries['unordered_edge_count']).fetchone()[0]
            ordered_vert_query = filtered_queries['ordered_vertices']
            ordered_edge_query = filtered_queries['ordered_edges']
        else:
            nv = conn.execute("SELECT COUNT(*) FROM VERTEX").fetchone()[0]
            ne = conn.execute(unordered_edge_count).fetchone()[0]
            ordered_vert_query = ordered_vertices
            ordered_edge_query = ordered_edges

        self.logger.info(f"Graph size: {nv:,} vertices and {ne:,} edges")
        self.graph_builder.init_builder(nv, ne)
        self.logger.info("Reading ordered graph...")
        self._read_graph(conn, ordered_vert_query, ordered_edge_query)
        self.logger.info("Graph reading completed")
        graph = self.graph_builder.build()

        # Save to cache if requested
        if use_cache:
            self.save_cache(graph)

        return graph

    @staticmethod
    def save_ordering(conn, ordering: Sequence[int]):
        conn.execute(create_ordering_table)
        conn.execute(create_ordering_index)
        # noinspection SqlWithoutWhere
        conn.execute("DELETE FROM ORDERING")
        for idx, vertex_id in enumerate(ordering):
            conn.execute(f"INSERT INTO ORDERING (id, position) values({vertex_id + 1}, {idx + 1})")
        conn.commit()

    def get_vertex_key(self) -> Dict[int, Tuple[str, str]]:
        """
        Get the "vertex key", a dictionary keyed by the vertex id (int) with values
        that are tuples of (external id, string designation)
        """
        conn = sl.connect(self.db_file)

        # Use filtered query if hop filtering is active
        if self.hops is not None and self._check_iteration_data(conn):
            filtered_queries = self._get_filtered_queries(self.hops)
            query = filtered_queries['vertex_key']
        else:
            query = vertex_key_query

        vertices = conn.execute(query)
        vertex_key = dict()
        for vertex in vertices:
            vertex_id = vertex[0] - 1
            external_id = vertex[1]
            vertex_name = f"'{vertex[3]}', '{vertex[2]}'"
            vertex_key[vertex_id] = (external_id, vertex_name)
        return vertex_key

    def _read_graph(self, conn, vertex_query, edge_query):
        # First pass: collect all vertex genders
        # This ensures gender info is available when edges are processed
        vertices_data = conn.execute(vertex_query).fetchall()
        for vertex in vertices_data:
            vertex_id = vertex[0] - 1
            color = vertex[1]
            self.graph_builder.add_gender(vertex_id, color)

        # Second pass: process vertices and edges in order
        vertices = iter(vertices_data)
        edges = conn.execute(edge_query)

        read_vertex = True
        read_edge = True
        i = color = src = dst = None
        edges_exhausted = False

        while True:
            if read_vertex:
                try:
                    vertex = next(vertices)
                    i = vertex[0] - 1
                    color = vertex[1]
                    read_vertex = False
                except StopIteration:
                    # No more vertices to read
                    if edges_exhausted:
                        break
                    pass
            if read_edge:
                try:
                    edge = next(edges)
                    src = edge[0] - 1
                    dst = edge[1] - 1
                    read_edge = False
                except StopIteration:
                    # we've consumed the last edge, set the src less than zero so that we can process
                    # remaining vertices
                    src = -1
                    edges_exhausted = True
                    read_edge = False
            if src is not None and i is not None and src < i:
                self.graph_builder.add_vertex(i, color)
                read_vertex = True
            elif src is not None and i is not None:
                self.graph_builder.add_edge(src, dst)
                read_edge = True
