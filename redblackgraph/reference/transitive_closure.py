from typing import Sequence
from .shortest_path import floyd_warshall
from redblackgraph.types.transitive_closure import TransitiveClosure

def transitive_closure(M: Sequence[Sequence[int]]) -> TransitiveClosure:
    return TransitiveClosure(*floyd_warshall(M))