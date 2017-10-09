from redblackgraph.simple import vec_avos, avos


def mat_avos_expanded(X, Y):
    result = [[0] * len(X)] * len(X)
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            elements = []
            for k in range(len(Y)):
                a, b = X[i][k], Y[k][j]
                if a > 0 and b > 0:
                    elements.append(avos(a, b))
            result[i][j] = min([e for e in elements if e > 0], default=0)
    return result


def mat_avos(A, B):
    '''Given two matrices, compute the "avos" product.'''
    return [[min([avos(a, b) for a, b in zip(A_row, B_col) if a > 0 and b > 0], default=0) for B_col in zip(*B)] for
            A_row in A]

if __name__ == '__main__':

    A = [[0, 2, 3, 0, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [2, 0, 0, 0, 1]]
    result = mat_avos_expanded(A, A)
    print(result)
    result = mat_avos(A, A)
    print(result)