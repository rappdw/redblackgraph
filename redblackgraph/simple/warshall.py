from redblackgraph.simple import generation
from redblackgraph.simple import avos


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
                if W[i][k] > 1 and W[k][j] > 1:
                    W[i][j] = min(W[i][j], avos(W[i][k], W[k][j])) if not W[i][j] == 0 else avos(W[i][k], W[k][j])
                else:
                    W[i][j] = W[i][j]
                diameter = max(diameter, W[i][j])
    return W, generation(diameter)