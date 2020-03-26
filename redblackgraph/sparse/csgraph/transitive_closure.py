from redblackgraph.sparse import rb_matrix
from redblackgraph.types.transitive_closure import TransitiveClosure
from redblackgraph.sparse.csgraph._shortest_path import shortest_path

def transitive_closure(R: rb_matrix) -> TransitiveClosure:
    return TransitiveClosure(*shortest_path(R, method="D", directed=True, overwrite=False))