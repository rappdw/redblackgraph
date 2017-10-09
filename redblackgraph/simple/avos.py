from redblackgraph.simple import generation


def avos(x, y):
    '''The avos product is a transitive relationship operator.
    Given that:
      - vertex a is related to vertex b by x
      - vertex b is related to vertex c by y
    This product should return the value indicating how a is related to c
    '''

    # The domain of this function is natural numbers
    if x <= 1 or y <= 1:
        # there are special cases if either operand is 0 or 1
        # n *avos* 0 = n if n is even
        # n *avos* 1 = n if n is odd
        # 0 otherwise
        return x if (y == 0 and x % 2 == 0) or (y == 1 and x % 2 == 1) else 0

    # There are some edge cases that need to be considered, namely what is meant
    generationNumber = generation(y)
    return (y & (2 ** generationNumber - 1)) | (x << generationNumber)
