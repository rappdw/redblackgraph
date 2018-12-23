"""RedBlack matrix"""

import numpy as np
from ._sparsetools import rbm_matmat_pass1, rbm_matmat_pass2
from redblackgraph.reference import Triangularization
from scipy.sparse.csr import csr_matrix
from scipy.sparse.sputils import (get_index_dtype, upcast)

class rb_matrix(csr_matrix):

    # format = 'rbm' # keep format as csr, aside from matmul (and specifically csr_matmat_pass2) all
    # other operations should be the same

    def __matmul__(self, other):
        return self._mul_sparse_matrix(other)

    def __rmatmul__(self, other):
        # convert to this format
        return self.__class__(other)._mul_sparse_matrix(self)

    def _mul_sparse_matrix(self, other):
        # this is lifted from scipy.sparse.compressed._mul_sparse_matrix and is identical aside from explicitely
        # using rbm_matmat_passx as the functions rather than looking them up from functions available
        # in the scipy._sparsetools module
        M, K1 = self.shape
        K2, N = other.shape

        major_axis = self._swap((M, N))[0]
        other = self.__class__(other)  # convert to this format

        idx_dtype = get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=M * N)
        indptr = np.empty(major_axis + 1, dtype=idx_dtype)

        rbm_matmat_pass1(M, N,
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

        rbm_matmat_pass2(M, N, np.asarray(self.indptr, dtype=idx_dtype),
                         np.asarray(self.indices, dtype=idx_dtype),
                         self.data,
                         np.asarray(other.indptr, dtype=idx_dtype),
                         np.asarray(other.indices, dtype=idx_dtype),
                         other.data,
                         indptr, indices, data)

        return self.__class__((data, indices, indptr), shape=(M, N))

    def vertex_relational_composition(self, u, v, c, compute_closure=False):
        '''
        Given simple row vector u, and simple column vector v where
        u, v represent a vertex, lambda, not currently represented in self, compose R_{lambda}
        which is the transitive closure for this graph with lambda included
        :param u: simple row vector for new vertex, lambda
        :param v: simple column vector for new vertex, lambda
        :param c: color of the new vertex, either -1 or 1
        :param compute_closure: if True, compute the closure of R prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda
        '''
        pass

    def edge_relational_composition(self, alpha, beta, relationship, compute_closure=False):
        '''
        Given simple two vertex indices, alpha and beta, along with the relationship ({2, 3}),
        compose R_{lambda} which is the transitive closure for this graph with the edge added
        :param alpha: index in self that is the source of relationship np
        :param beta: index in self that is the targe of relationship np
        :param relationship: r(alpha, beta)
        :param compute_closure: if True, compute the closure of R prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda, new_diameter
        '''
        pass

    def triangularize(self) -> Triangularization:
        pass

    def rc(self, u, v, color):
        '''
        rc or Relational Composition is the mechanism used to add a new node
        into a Red/Black Graph. (See README.md)

        Keyword arguments:
        u -- *simple* row vector
        v -- *simple* column vector
        color -- the coloring of the new node
        '''
        pass