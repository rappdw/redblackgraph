from redblackgraph.reference.avos import avos_sum, avos_product
from redblackgraph.reference.mat_avos import mat_avos
import copy


def vertex_relational_composition(u, R, v, color):
    '''
    Given simple row vector u, transitively closed matrix R, and simple column vector v where
    u and v represent a vertex, lambda, not currently represented in R, compose R_{lambda}
    which is the transitive closure for the graph with lambda included
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
            if uc_lambda[0][j] != 0:
                R_lambda[i][j] = avos_sum(avos_product(vc_lambda[i][0], uc_lambda[0][j]), R_lambda[i][j])
    R_lambda[N].append(color)
    return R_lambda

def edge_relational_composition(R, alpha, beta, relationship):
    '''
    Given a transitively closed graph, two vertices in that graph, alpha and beta, and the
    relationship from alpha to beta, compose R'', which is the transitive closure with the
    new edge included
    :param R:
    :param alpha: a vertex in the graph (row index)
    :param beta: a vertex in the grpah (column index)
    :param relationship: r(alpha, beta)
    :return: transitive closure of the grpah, R, with new edge
    '''
    N = len(R)
    u_lambda = [R[alpha]]
    u_lambda[0][beta] = relationship
    u_lambda = mat_avos(u_lambda, R)
    R_lambda = copy.deepcopy(R)
    R_lambda[alpha] = u_lambda[0]
    for i in range(N):
        for j in range(N):
            if R_lambda[alpha][j] != 0:
                R_lambda[i][j] = avos_sum(avos_product(R_lambda[i][alpha], R_lambda[alpha][j]), R_lambda[i][j])
    return R_lambda
