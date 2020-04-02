import cython
import numpy as np

cimport numpy as np

include 'parameters.pxi'
include '_rbg_math.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _vertex_compose_pass1(DTYPE_t[:] u, DTYPE_t[:, :] A, DTYPE_t[:] v, DTYPE_t[:] ul, DTYPE_t[:] vl):
    cdef unsigned int N = A.shape[0]
    for i in range(N):
        for j in range(N):
            ul[i] = avos_sum(ul[i], avos_product(u[j], A[j][i]))
            vl[i] = avos_sum(vl[i], avos_product(A[i][j], v[j]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _vertex_compose_pass2(DTYPE_t[:] u, DTYPE_t[:, :] A, DTYPE_t[:] v, DTYPE_t color):
    cdef unsigned int N = A.shape[0]
    R = np.zeros((N + 1, N + 1), dtype=DTYPE)
    cdef DTYPE_t[:, :] Rm = R
    for i in range(N):
        for j in range(N):
            if u[j] != 0:
                Rm[i, j] = avos_sum(avos_product(v[i], u[j]), A[i][j])
            else:
                Rm[i, j] = A[i][j]
    for i in range(N):
        Rm[i, N] = v[i]
        Rm[N, i] = u[i]
    Rm[N, N] = color
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
    cdef unsigned int N = R.shape[0]

    cdef DTYPE_t[:, :] Rm = R
    cdef DTYPE_t[:, :] Rm_lambda
    cdef DTYPE_t[:] um = u.reshape((N,))
    cdef DTYPE_t[:] vm = v.reshape((N,))
    cdef DTYPE_t[:] uc_lambda, vc_lambda

    assert N == R.shape[1]

    uc_lambda = np.zeros(N, dtype=DTYPE)
    vc_lambda = np.zeros(N, dtype=DTYPE)

    _vertex_compose_pass1(um, Rm, vm, uc_lambda, vc_lambda)
    return _vertex_compose_pass2(uc_lambda, Rm, vc_lambda, color)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t[:] _edge_compose_pass1(DTYPE_t[:] u, DTYPE_t[:, :] A):
    cdef unsigned int N = A.shape[0]
    cdef DTYPE_t[:] ul = np.zeros(N, dtype=DTYPE)
    for i in range(N):
        for j in range(N):
            ul[i] = avos_sum(ul[i], avos_product(u[j], A[j][i]))
    return ul


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _edge_compose_pass2(unsigned int alpha, DTYPE_t[:] u, DTYPE_t[:, :] A):
    cdef unsigned int N = A.shape[0]
    R = np.zeros((N, N), dtype=DTYPE)
    cdef DTYPE_t[:, :] Rm = R
    for i in range(N):
        for j in range(N):
            if i == alpha:
                Rm[i][j] = u[j]
            elif u[j] != 0:
                Rm[i][j] = avos_sum(avos_product(A[i][alpha], u[j]), A[i][j])
            else:
                Rm[i][j] = A[i][j]
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
    cdef unsigned int N = R.shape[0]
    assert N == R.shape[1]

    cdef DTYPE_t[:, :] Rm = R

    Rm[alpha, beta], tmp = relationship, Rm[alpha, beta]
    cdef DTYPE_t[:] u_lambda = _edge_compose_pass1(Rm[alpha], R)
    R_lambda = _edge_compose_pass2(alpha, u_lambda, Rm)
    Rm[alpha, beta] = tmp
    return R_lambda
