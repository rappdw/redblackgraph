from redblackgraph.simple import avos
from redblackgraph.simple.util import nz_min

def vec_avos(x, y):
    '''Given two vectors, compute the "avos" dot product.'''
    return nz_min([avos(a, b) for a, b in zip(x, y)])
