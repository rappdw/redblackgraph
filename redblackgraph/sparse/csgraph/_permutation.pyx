import numpy as np
cimport numpy as np
cimport cython

include 'parameters.pxi'
include '_rbg_math.pxi'


def permute(A, p, assume_upper_triangular=False):
    '''Permutes an input matrix based on the vertex ordering specified.

    Equivalent to P * A * P-1 (where P is a permutation of the identity matrix specified by p)
    '''
    cdef np.ndarray B = np.zeros(A.shape, dtype=DTYPE)
    _permute(A, B, p, assume_upper_triangular)
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t _permute(np.ndarray[DTYPE_t, ndim=2, mode='c'] A, np.ndarray[DTYPE_t, ndim=2, mode='c'] B, np.ndarray[ITYPE_t, ndim=1, mode='c'] p, bint assume_upper_triangular):
    cdef unsigned int i, j, start, N = B.shape[0]
    assert B.shape[1] == N
    assert p.shape[0] == N
    cdef DTYPE_t[:, :] Am = A
    cdef DTYPE_t[:, :] Bm = B
    cdef ITYPE_t[:] pm = p

    for i in range(N):
        if assume_upper_triangular:
            start = i
        else:
            start = 0

        for j in range(start, N):
            Bm[i][j] = Am[pm[i]][pm[j]]