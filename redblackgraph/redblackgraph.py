import numpy as np
from scipy.sparse.base import isspmatrix
from scipy.sparse import csr_matrix
from scipy.sparse import _sparsetools


def generation(x):
    if x < 0:
        raise ValueError('Undefined when x < 0. x: {}'.format(x))
    generation = 0
    while x > 0:
        x >>= 1
        generation += 1
    return generation - 1 if generation > 0 else generation


def avos(a, b):
    generationNumber = generation(b)
    if a == 0 or a == 1:
        if generationNumber == 0 and a != b:
            raise ValueError('Undefined avos. a: {}, b: {}'.format(a, b))
        return b
    return (b & (pow(2, generationNumber) - 1)) | (a << generationNumber)


def _csr_matmat_pass1(n_row, n_col, Ap, Aj, Bp, Bj, Cp):
    mask = np.empty(n_col, dtype=np.int32)
    mask.fill(-1)
    Cp[0] = 0
    nnz = 0

    for i in range(0, n_row):
        row_nnz = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1
        next_nnz = nnz + row_nnz
        nnz = next_nnz
        Cp[i + 1] = nnz
    pass

def accumulate_cell_value(cell, transitive_relationship):
    return transitive_relationship if cell == 0 else min(cell, transitive_relationship)

def _csr_matmat_pass2(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx):
    next = np.empty(n_col, dtype=np.int32)
    next.fill(-1)
    sums = np.zeros(n_col, dtype=np.int32)
    nnz = 0
    Cp[0] = 0

    for i in range(0, n_row):
        head = -2
        length = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            v = Ax[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                sums[k] = accumulate_cell_value(sums[k], avos(v, Bx[kk]))
                if next[k] == -1:
                    next[k] = head
                    head = k
                    length += 1
        for jj in range(0, length):
            if sums[head] != 0 or head == i:  # only take non-zero elements unless it's on the diagonal
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1
            temp = head
            head = next[head]
            next[temp] = -1
            sums[temp] = 0
        Cp[i + 1] = nnz


def expand(m):
    M, N = m.shape
    if M != N:
        raise ValueError('Input must be a 2 dimensional square matrix')
    if not isspmatrix(m):
        raise ValueError('Currently only available for sparse matrices')
    m = m.tocsr()

    major_axis = m.shape[0]
    idx_dtype = np.int32

    indptr = np.empty(major_axis + 1, dtype=idx_dtype)

    fn = getattr(_sparsetools, m.format + '_matmat_pass1')
    #fn = _csr_matmat_pass1

    fn(M, N,
       np.asarray(m.indptr, dtype=idx_dtype),
       np.asarray(m.indices, dtype=idx_dtype),
       np.asarray(m.indptr, dtype=idx_dtype),
       np.asarray(m.indices, dtype=idx_dtype),
       indptr)

    nnz = indptr[-1]
    indptr = np.asarray(indptr, dtype=idx_dtype)
    indices = np.zeros(nnz, dtype=idx_dtype)
    data = np.zeros(nnz, dtype=idx_dtype)

    fn = _csr_matmat_pass2
    fn(M, N,
       np.asarray(m.indptr, dtype=idx_dtype),
       np.asarray(m.indices, dtype=idx_dtype),
       m.data,
       np.asarray(m.indptr, dtype=idx_dtype),
       np.asarray(m.indices, dtype=idx_dtype),
       m.data,
       indptr, indices, data)

    result = csr_matrix((data, indices, indptr), shape=(M, N))
    return result
