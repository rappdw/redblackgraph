def generation(pedigree_number):
    '''Given a pedigree_number, representing a relationship from a "root" vertex to an "ancester" vertex,
    return the number of edges that must be followed in order to "walk" from the "root" to the "ancester".'''
    x = pedigree_number
    gen = 0
    while (x > 1):
        x >>= 1
        gen += 1
    return gen


def get_traversal_path(pedigree_number):
    '''Given a pedigree_number, representing a relationship from a "root" vertex to an "ancester" vertex,
    return the traversal path of edges to red or black vertices to "walk" from the "root" to the "ancesster".

    For example, input of 14 results in ['b', 'b', 'r'] which indicates that starting at the "root" vertex, follow
    the edge to the black vertex, then the edge to the black vertex then the edge to the red vertex.'''
    x = pedigree_number
    path = []
    mask = 1
    while (x > 1):
        path.insert(0, 'b') if x & mask else path.insert(0, 'r')
        x >>= 1
    return path


def avos(x, y):
    '''The avos product is a transitive relationship operator.
    Given that:
      - vertex a is related to vertex b by x
      - vertex b is related to vertex c by y
    This product should return the value indicating how a is related to c
    '''

    # The domain of this function is all positive integers greater than 1
    if x <= 1 or y <= 1:
        raise ValueError(f'avos({x}, {y}) is not defined')

    # There are some edge cases that need to be considered, namely what is meant
    generationNumber = generation(y)
    return (y & (2 ** generationNumber - 1)) | (x << generationNumber)


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