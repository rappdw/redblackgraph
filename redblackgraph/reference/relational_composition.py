import numpy as np
from redblackgraph.reference.rbg_math import avos_sum, avos_product


def _vertex_compose_pass1(u, A, v):
    N = len(A)
    ul = np.zeros(N, dtype=np.int32)
    vl = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            ul[i] = avos_sum(ul[i], avos_product(u[0][j], A[j][i]))
            vl[i] = avos_sum(vl[i], avos_product(A[i][j], v[j][0]))
    return ul, vl


def _vertex_compose_pass2(u, A, v, color):
    N = len(A)
    R = np.zeros((N + 1, N + 1), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if u[j] != 0:
                R[i, j] = avos_sum(avos_product(v[i], u[j]), A[i][j])
            else:
                R[i, j] = A[i][j]
    for i in range(N):
        R[i, N] = v[i]
        R[N, i] = u[i]
    R[N, N] = color
    return R


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
    uc_lambda, vc_lambda = _vertex_compose_pass1(u, R, v)
    return _vertex_compose_pass2(uc_lambda, R, vc_lambda, color)


def _edge_compose_pass1(u, A):
    N = len(A)
    ul = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            ul[i] = avos_sum(ul[i], avos_product(u[0][j], A[j][i]))
    return ul


def _edge_compose_pass2(alpha, u, A):
    N = len(A)
    R = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == alpha:
                R[i][j] = u[j]
            elif u[j] != 0:
                R[i][j] = avos_sum(avos_product(A[i][alpha], u[j]), A[i][j])
            else:
                R[i][j] = A[i][j]
    return R


def edge_relational_composition(R, alpha, beta, relationship):
    '''
    Given a transitively closed graph, two vertices in that graph, alpha and beta, and the
    relationship from alpha to beta, compose R', which is the transitive closure with the
    new edge included
    :param R:
    :param alpha: a vertex in the graph (row index)
    :param beta: a vertex in the grpah (column index)
    :param relationship: r(alpha, beta)
    :return: transitive closure of the grpah, R, with new edge
    '''
    N = len(R)
    R[alpha][beta], tmp = relationship, R[alpha][beta]
    u_lambda = _edge_compose_pass1([R[alpha]], R)
    R_lambda = _edge_compose_pass2(alpha, u_lambda, R)
    R[alpha][beta] = tmp
    return R_lambda
