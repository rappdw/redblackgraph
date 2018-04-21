import numpy as np
from dataclasses import dataclass
from typing import Sequence
from redblackgraph.reference import avos_sum, avos_product, leftmost_significant_bit_position


@dataclass
class WarshallResult:
    W: np.array
    diameter: int

def warshall(M: Sequence[Sequence[int]]) -> WarshallResult:
    '''Computes the transitive closure of a Red Black adjacency matrix and as a side-effect,
    the diameter.'''

    # Modification of stardard warshall algorithm:
    # * Replaces innermost loop's: `W[i][j] = W[i][j] or (W[i][k] and W[k][j])`
    # * Adds diameter calculation
    n = len(M)
    W = np.array(M, copy=True)
    diameter = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))
                diameter = max(diameter, W[i][j])
    return WarshallResult(W, leftmost_significant_bit_position(diameter))