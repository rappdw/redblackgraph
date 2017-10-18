from redblackgraph.simple import generation


def avos(x, y):
    '''The avos product is a transitive relationship operator.
    Given that:
      - vertex a is related to vertex b by x
      - vertex b is related to vertex c by y
    This product should return the value indicating how a is related to c
    '''

    # An interesting case is presented if x or y is 1 or -1. In these cases the avos product
    # represents the situation where a and b are the same vertex or b and c are the same vertex.
    # In a sense, this is the avos identity
    if x <= 1 or y <= 1:
        return 0 if x == 0 or y == 0 else x if y == 1 or y == -1 else y

    generationNumber = generation(y)
    return (y & (2 ** generationNumber - 1)) | (x << generationNumber)
