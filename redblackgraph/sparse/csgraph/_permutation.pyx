import numpy as np
cimport numpy as np

include 'parameters.pxi'
include '_rbg_math.pxi'


def permute(A, p, assume_upper_triangular=False):
    '''Permutes an input matrix based on the vertex ordering specified.

    Equivalent to P * A * P-1 (where P is a permutation of the identity matrix specified by p)
    '''
    cdef np.ndarray B = np.zeros(A.shape, dtype=DTYPE)
    _permute(A, B, p, assume_upper_triangular)
    return B

cdef DTYPE_t _permute(np.ndarray[DTYPE_t, ndim=2, mode='c'] A, np.ndarray[DTYPE_t, ndim=2, mode='c'] B, np.ndarray[ITYPE_t, ndim=1, mode='c'] p, bint assume_upper_triangular):
    cdef unsigned int i, j, start, N = B.shape[0]
    assert B.shape[1] == N
    assert p.shape[0] == N

    for i in range(N):
        if assume_upper_triangular:
            start = i
        else:
            start = 0

        for j in range(start, N):
            B[i][j] = A[p[i]][p[j]]