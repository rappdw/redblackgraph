import numpy as np
from typing import Sequence
from .avos import avos_sum, avos_product, MSB
from redblackgraph.types.transitive_closure import TransitiveClosure


def transitive_closure(M: Sequence[Sequence[int]], copy:bool=True) -> TransitiveClosure:
    '''Computes the transitive closure of a Red Black adjacency matrix and as a side-effect,
    the diameter.'''

    # Modification of stardard warshall algorithm:
    # * Replaces innermost loop's: `W[i][j] = W[i][j] or (W[i][k] and W[k][j])`
    # * Adds diameter calculation
    n = len(M)
    W = np.array(M, copy=copy)
    diameter = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))
                diameter = max(diameter, W[i][j])
    return TransitiveClosure(W, MSB(diameter))