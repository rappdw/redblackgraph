"""RedBlack matrix"""

import numpy as np
from scipy.sparse.csr import csr_matrix
import _sparsetools
from scipy.sparse.sputils import (get_index_dtype, upcast)


class rb_matrix(csr_matrix):

    #format = 'rbm' # keep format as csr, aside from matmul all other operations should be the same

    def _mul_sparse_matrix(self, other):
        # this is lifted from scipy.csr._mul_sparxe_matrix and is identical asside from the
        # _sparsetools reference and explicitely using '_rbm_matmat_...' in the getattr
        M, K1 = self.shape
        K2, N = other.shape

        major_axis = self._swap((M,N))[0]
        other = self.__class__(other)  # convert to this format

        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=M*N)
        indptr = np.empty(major_axis + 1, dtype=idx_dtype)

        fn = getattr(_sparsetools, 'rbm_matmat_pass1')
        fn(M, N,
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           indptr)

        nnz = indptr[-1]
        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=nnz)
        indptr = np.asarray(indptr, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        fn = getattr(_sparsetools, 'rbm_matmat_pass2')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        return self.__class__((data,indices,indptr),shape=(M,N))

