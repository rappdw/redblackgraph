from redblackgraph.simple import avos_product, avos_sum

def vec_avos(u, v):
    '''Given two vectors, compute the avos product.'''
    return avos_sum([avos_product(a, b) for a, b in zip(u, v)])
