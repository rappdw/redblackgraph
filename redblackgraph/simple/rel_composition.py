from redblackgraph.simple import avos, mat_avos
from redblackgraph.simple.util import nz_min
import copy


def relational_composition(u, A, v):
    '''
    Given simple row vector u, transitively closed matrix A, and simple column vector v where
    u and v represent a vertex, lambda, not currently represented in A, compose A_{\lambda} wich is
    the transitive closure for the graph with lambda included
    :param u: simple row vector for new vertex, lambda
    :param A: transitive closure for Red Black graph
    :param v: simple column vector for new vertex, lambda
    :return: transitive closure for Red BLack graph with lambda
    '''
    N = len(A)
    uc_lambda = mat_avos([u[0][:-1]], A)
    vc_lambda = mat_avos(A, v[:-1])
    A_lambda = copy.deepcopy(A)
    A_lambda.append(uc_lambda[0])
    for i in range(N):
        A_lambda[i].append(vc_lambda[i][0])
        for j in range(N):
            if not uc_lambda[0][j] == 0:
                A_lambda[i][j] = nz_min(avos(vc_lambda[i][0], uc_lambda[0][j]), A_lambda[i][j])
    A_lambda[N].append(u[0][N])
    return A_lambda
