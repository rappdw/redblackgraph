from redblackgraph.simple import avos
from redblackgraph.simple.util import nz_min

def vec_avos(u, v):
    '''Given two vectors, compute the avos product.'''
    return nz_min([avos(a, b) for a, b in zip(u, v)])
