from infix import shift_infix as infix


def generation(x):
    if x < 0:
        raise ValueError('Undefined when x < 0. x: {}'.format(x))
    generation = 0
    while x > 0:
        x >>= 1
        generation += 1
    return generation - 1 if generation > 0 else generation


@infix
def avos(a, b):
    generationNumber = generation(b)
    if a == 0 or a == 1:
        if generationNumber == 0 and a != b:
            raise ValueError('Undefined avos. a: {}, b: {}'.format(a, b))
        return b
    return (b & (pow(2, generationNumber) - 1)) | (a << generationNumber)


@infix
def acc(cell, transitive_relationship):
    return transitive_relationship if cell == 0 else min(cell, transitive_relationship)

