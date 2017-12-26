from redblackgraph.simple import avos, mat_avos
from redblackgraph.simple.util import nz_min
import copy


def relational_composition(u, R, v, color):
    '''
    Given simple row vector u, transitively closed matrix R, and simple column vector v where
    u and v represent a vertex, lambda, not currently represented in R, compose R_{\lambda} wich is
    the transitive closure for the graph with lambda included
    :param u: simple row vector for new vertex, lambda
    :param R: transitive closure for Red Black graph
    :param v: simple column vector for new vertex, lambda
    :param color: color of the node either -1 or 1
    :return: transitive closure of the graph, R, with new node, lambda
    '''
    N = len(R)
    uc_lambda = mat_avos(u, R)
    vc_lambda = mat_avos(R, v)
    R_lambda = copy.deepcopy(R)
    R_lambda.append(uc_lambda[0])
    for i in range(N):
        R_lambda[i].append(vc_lambda[i][0])
        for j in range(N):
            if not uc_lambda[0][j] == 0:
                R_lambda[i][j] = nz_min(avos(vc_lambda[i][0], uc_lambda[0][j]), R_lambda[i][j])
    R_lambda[N].append(color)
    return R_lambda
