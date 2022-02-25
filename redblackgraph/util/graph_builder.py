from typing import Sequence

import numpy as np
import redblackgraph as rb

from fscrawler import AbstractGraphBuilder, RelationshipDbReader
from scipy.sparse import coo_matrix
from redblackgraph.sparse.csgraph import avos_canonical_ordering


class RbgGraphBuilder(AbstractGraphBuilder):

    def __init__(self, sparse_threshold: int = 1000):
        super().__init__(sparse_threshold)
        self.idx = None
        self.val = None
        self.row = None
        self.col = None
        self.genders = None
        self.graph = None

    def init_builder(self, nv: int, ne: int):
        if nv > self.sparse_threshold:
            self.graph = None
            self.val = np.zeros(ne + nv, dtype=np.int64)
            self.row = np.zeros(ne + nv, dtype=np.int64)
            self.col = np.zeros(ne + nv, dtype=np.int64)
            self.genders = np.zeros(nv, dtype=np.int64)
            self.idx = ne + nv - 1
        else:
            self.idx = None
            self.val = None
            self.row = None
            self.col = None
            self.genders = None
            self.graph = rb.matrix(np.zeros((nv, ne), dtype=np.int64))

    def get_ordering(self) -> Sequence[int]:
        if not self.graph:
            self.graph = rb.rb_matrix(coo_matrix((self.val, (self.row, self.col))))
        ordering = avos_canonical_ordering(self.graph.transitive_closure().W)
        return ordering.label_permutation

    def add_vertex(self, vertex_id: int, color: int):
        if self.graph:
            self.graph[vertex_id, vertex_id] = color
        else:
            self.val[self.idx] = color
            self.row[self.idx] = vertex_id
            self.col[self.idx] = vertex_id
            self.idx -= 1

    def add_edge(self, source_id: int, dest_id: int):
        if self.graph:
            self.graph[source_id, dest_id] = 3 if self.graph[dest_id, dest_id] == 1 else 2
        else:
            self.val[self.idx] = 3 if self.genders[dest_id] == 1 else 2
            self.row[self.idx] = source_id
            self.col[self.idx] = dest_id
            self.idx -= 1

    def add_gender(self, vertex_id: int, color: int):
        if not self.graph:
            self.genders[vertex_id] = color

    def build(self):
        if self.graph:
            rtn_graph = self.graph
        else:
            rtn_graph = rb.rb_matrix(coo_matrix((self.val, (self.row, self.col))))
            self.val = None
            self.row = None
            self.col = None
            self.genders = None
        return rtn_graph


if __name__ == "__main__":
    import time
    rdr = RelationshipDbReader("/Users/drapp/data/rbg/rappdw.db", 4, RbgGraphBuilder())
    g = rdr.read()
    start = time.time()
    closure = g.transitive_closure()
    duration = time.time() - start
    print(f"duration: {duration}")
