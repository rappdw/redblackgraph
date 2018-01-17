import numpy as np
from redblackgraph.simple import avos, generation
from .util import nz_min


def warshall(M):
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
                # output = nz_min(output, avos(ipR1, ipR2))
                W[i][j] = nz_min(W[i][j], avos(W[i][k], W[k][j]))
                diameter = max(diameter, W[i][j])
                # advance column of output
                # advance column of ipR2
            # advance row of output
            # advance row of ipR1
            # reset to coumn 0 of ipR2
        # reset output
        # reset ipR1 to row 0, and column k
        # advance row of ipR2, reset to column 0
    return W, generation(diameter)