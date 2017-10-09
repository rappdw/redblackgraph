from redblackgraph.simple import avos, generation
from .util import nz_min


def warshall(M):
    '''Computes the transitive closure of a Red Black adjacency matrix and as a side-effect,
    the diameter.'''

    # Modification of stardard warshall algorithm:
    # * Replaces innermost loop's: `W[i][j] = W[i][j] or (W[i][k] and W[k][j])`
    # * Adds diameter calculation
    n = len(M)
    W = M
    diameter = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i][j] = nz_min(W[i][j], avos(W[i][k], W[k][j]))
                diameter = max(diameter, W[i][j])
    return W, generation(diameter)