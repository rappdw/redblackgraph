import numpy as np
from typing import Sequence
from redblackgraph.reference.avos import avos_sum, avos_product
from redblackgraph.reference.util import MSB
from redblackgraph.types.transitive_closure import TransitiveClosure


def transitive_closure(M: Sequence[Sequence[int]], copy:bool=True) -> TransitiveClosure:
    '''Computes the transitive closure of a Red Black adjacency matrix and the diameter.'''

    # Modification of stardard warshall algorithm:
    #  - Replaces innermost loop's: `W[i][j] = avos_sum(W[i][j], avos_product(W[i][k],W[k][j]))`
    #  - Adds diameter calculation
    #  - Adds cycle detection

    n = len(M)
    W = np.array(M, copy=copy)
    diameter = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                product = avos_product(W[i][k], W[k][j])
                if i == j and not product in [-1, 0, 1]:
                    raise ValueError(f"Error: cycle detected! Vertex {i} has a path to itself. A({i},{k})={W[i][k]}, A({k},{j})={W[k][j]}")
                W[i][j] = avos_sum(W[i][j], product)
                diameter = max(diameter, W[i][j])
    return TransitiveClosure(W, MSB(diameter))