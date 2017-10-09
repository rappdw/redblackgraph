from redblackgraph.simple import avos
from redblackgraph.simple.util import nz_min


def mat_avos(A, B):
    '''Given two matrices, compute the "avos" product.'''
    return [[nz_min([avos(a, b) for a, b in zip(A_row, B_col)]) for B_col in zip(*B)] for A_row in A]
