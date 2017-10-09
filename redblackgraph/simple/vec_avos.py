from redblackgraph.simple import avos


def vec_avos(x, y):
    '''Given two vectors, compute the "avos" dot product.'''
    return min([avos(a, b) for a, b in zip(x, y) if a > 0 and b > 0], default=0)

if __name__ == '__main__':

    u = [0, 2, 3, 0, 0]
    v = [2, 0, 0, 0, 0]
    result = vec_avos(u, v)
    print(result)
