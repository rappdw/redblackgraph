from redblackgraph.simple import avos


def vec_avos(x, y):
    '''Given two vectors, compute the "avos" dot product.'''
    return min([e for e in [avos(a, b) for a, b in zip(x, y)] if e > 0], default=0)
