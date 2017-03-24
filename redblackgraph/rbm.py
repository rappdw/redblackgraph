"""RedBlack matrix"""

import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse.base import isspmatrix
from scipy.sparse import _sparsetools
from .operators import acc, avos


# this differs from the _csr_matmat_pass2 in scipy in two ways noted below in 1) and 2)
def _rbm_matmat_pass2(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx):
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
                # 1) rather than multiplication and summation, we perform the avos operator
                # in place of multiplication and accumulate_cell_value in place of
                # summation
                sums[k] = sums[k] << acc >> (v << avos >> Bx[kk])
                if next[k] == -1:
                    next[k] = head
                    head = k
                    length += 1
        for jj in range(0, length):
            # 2) We preserve zero element results if they are on the diagonal
            # perhaps another argument for 1 == self (see comment in operators.generation2
            if sums[head] != 0 or head == i:
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1
            temp = head
            head = next[head]
            next[temp] = -1
            sums[temp] = 0
        Cp[i + 1] = nnz


class rb_matrix(csr_matrix):

    def _mul_sparse_matrix(self, other):
        M, N = other.shape
        if M != N:
            raise ValueError('Input must be a 2 dimensional square matrix')
        if not isspmatrix(other):
            raise ValueError('Currently only available for sparse matrices')
        m = other.tocsr()

        major_axis = m.shape[0]
        idx_dtype = np.int32

        indptr = np.empty(major_axis + 1, dtype=idx_dtype)

        fn = getattr(_sparsetools, m.format + '_matmat_pass1')

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

        fn = _rbm_matmat_pass2
        fn(M, N,
           np.asarray(m.indptr, dtype=idx_dtype),
           np.asarray(m.indices, dtype=idx_dtype),
           m.data,
           np.asarray(m.indptr, dtype=idx_dtype),
           np.asarray(m.indices, dtype=idx_dtype),
           m.data,
           indptr, indices, data)

        return self.__class__((data,indices,indptr),shape=(M,N))
