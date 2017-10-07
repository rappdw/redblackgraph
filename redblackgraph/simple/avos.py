from redblackgraph.simple import generation

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
