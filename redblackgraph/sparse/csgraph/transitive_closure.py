from redblackgraph.sparse import rb_matrix
from redblackgraph.types.transitive_closure import TransitiveClosure
from redblackgraph.sparse.csgraph._shortest_path import shortest_path

def transitive_closure(R: rb_matrix, method="D") -> TransitiveClosure:
    return TransitiveClosure(*shortest_path(R, method=method, directed=True, overwrite=False))

def transitive_closure_floyd_warshall(R: rb_matrix) -> TransitiveClosure:
    return transitive_closure(R, method="FW")

def transitive_closure_dijkstra(R: rb_matrix) -> TransitiveClosure:
    return transitive_closure(R, method="D")