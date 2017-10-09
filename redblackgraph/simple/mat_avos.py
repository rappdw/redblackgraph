from redblackgraph.simple import avos


def mat_avos(A, B):
    '''Given two matrices, compute the "avos" product.'''
    return [[min([avos(a, b) for a, b in zip(A_row, B_col) if not a == 0 and not b == 0], default=0) for B_col in zip(*B)] for A_row in A]
