from redblackgraph.simple import avos_product, avos_sum


def mat_avos(A, B):
    '''Given two matrices, compute the "avos" product.'''
    return [[avos_sum([avos_product(a, b) for a, b in zip(A_row, B_col)]) for B_col in zip(*B)] for A_row in A]
