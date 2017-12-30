from redblackgraph.simple import generation


def avos(x, y):
    '''The avos product is the transitive relationship function.
    Given that:
      - vertex a is related to vertex b by x
      - vertex b is related to vertex c by y
    return the avos product which is how vertex a is related to vertex c
    '''

    # The zero property of the avos product; the transitive relationship is 0 if
    # either vertex a to vertex b is zero, or vertex b to vertex c is zero
    if x == 0 or y == 0:
        return 0

    # The identity property of the avos product; the relationship of 1 or -1 is a
    # "self relationship". If either the relationship of a to b or b to c is self,
    # then the transitive relationship is the non-self relationship
    if x <= 1 or y <= 1:
        self_relationship, non_self_relationship = (x, y) if x <= 1 else (y, x)
        return non_self_relationship

    generationNumber = generation(y)
    return (y & (2 ** generationNumber - 1)) | (x << generationNumber)
