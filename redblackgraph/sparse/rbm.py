"""RedBlack matrix"""

import numpy as np
from redblackgraph.sparse._sparsetools import rbm_matmat_pass1, rbm_matmat_pass2
from scipy.sparse import csr_matrix, get_index_dtype
from scipy.sparse._sputils import upcast


class rb_matrix(csr_matrix):

    # format = 'rbm' # keep format as csr, aside from matmul (and specifically csr_matmat_pass2) all
    # other operations should be the same

    def __matmul__(self, other):
        return self._mul_sparse_matrix(other)

    def __rmatmul__(self, other):
        # convert to this format
        return self.__class__(other)._mul_sparse_matrix(self)

    def __mul__(self, other):
        # Override * operator to use AVOS multiplication for sparse matrices
        # But allow scalar multiplication to use parent implementation
        if np.isscalar(other):
            return super().__mul__(other)
        return self._mul_sparse_matrix(other)

    def __rmul__(self, other):
        # Handle scalar multiplication
        if np.isscalar(other):
            return super().__rmul__(other)
        # convert to this format for matrix multiplication
        return self.__class__(other)._mul_sparse_matrix(self)

    def transitive_closure(self, method="D"):
        from .csgraph import transitive_closure
        return transitive_closure(self, method)

    def _mul_sparse_matrix(self, other):
        # this is lifted from scipy.sparse.compressed._mul_sparse_matrix and is identical aside from explicitely
        # using rbm_matmat_passx as the functions rather than looking them up from functions available
        # in the scipy._sparsetools module
        M, K1 = self.shape
        K2, N = other.shape

        major_axis = self._swap((M, N))[0]
        other = self.__class__(other)  # convert to this format
        
        # Ensure matrices are in canonical form (sorted indices, no duplicates)
        # This prevents segfaults in the C++ code when matrices have been modified in place
        if not self.has_sorted_indices:
            self.sort_indices()
        self.sum_duplicates()
        if not other.has_sorted_indices:
            other.sort_indices()
        other.sum_duplicates()

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
        out_dtype = np.dtype(upcast(self.dtype, other.dtype))

        # Promote to 64-bit for safety when calling the AVOS C++ kernels.
        # For unsigned inputs, note that RED_ONE is represented as the *max*
        # value of the dtype (e.g. uint8(255)). When upcasting to uint64 we must
        # remap that sentinel to uint64 max, otherwise the kernel won't recognize
        # it as RED_ONE.
        data_dtype = out_dtype
        if out_dtype.kind == 'u' and out_dtype.itemsize < 8:
            data_dtype = np.dtype(np.uint64)
        elif out_dtype.kind == 'i' and out_dtype.itemsize < 8:
            data_dtype = np.dtype(np.int64)
        data = np.empty(nnz, dtype=data_dtype)

        A_data = self.data
        B_data = other.data

        if np.dtype(A_data.dtype) != data_dtype:
            A_data = np.asarray(A_data, dtype=data_dtype)
            if out_dtype.kind == 'u' and data_dtype.kind == 'u':
                orig_max = np.iinfo(self.data.dtype).max
                tgt_max = np.iinfo(data_dtype).max
                A_data[A_data == orig_max] = tgt_max

        if np.dtype(B_data.dtype) != data_dtype:
            B_data = np.asarray(B_data, dtype=data_dtype)
            if out_dtype.kind == 'u' and data_dtype.kind == 'u':
                orig_max = np.iinfo(other.data.dtype).max
                tgt_max = np.iinfo(data_dtype).max
                B_data[B_data == orig_max] = tgt_max

        rbm_matmat_pass2(M, N, np.asarray(self.indptr, dtype=idx_dtype),
                         np.asarray(self.indices, dtype=idx_dtype),
                         A_data,
                         np.asarray(other.indptr, dtype=idx_dtype),
                         np.asarray(other.indices, dtype=idx_dtype),
                         B_data,
                         indptr, indices, data)

        result = rb_matrix((data, indices, indptr), shape=(M, N))

        # If we used a wider dtype for safe computation, cast the result back to
        # the expected (upcast) output dtype so downstream comparisons and APIs
        # behave consistently.
        if result.dtype != out_dtype:
            result = rb_matrix((result.data.astype(out_dtype, copy=False), result.indices, result.indptr), shape=result.shape)
        return result
    #
    # def vertex_relational_composition(self, u, v, c, compute_closure=False):
    #     '''
    #     Given simple row vector u, and simple column vector v where
    #     u, v represent a vertex, lambda, not currently represented in self, compose R_{lambda}
    #     which is the transitive closure for this graph with lambda included
    #     :param u: simple row vector for new vertex, lambda
    #     :param v: simple column vector for new vertex, lambda
    #     :param c: color of the new vertex, either -1 or 1
    #     :param compute_closure: if True, compute the closure of R prior to performing the relational composition
    #     :return: transitive closure for Red BLack graph with lambda
    #     '''
    #     pass
    #
    # def edge_relational_composition(self, alpha, beta, relationship, compute_closure=False):
    #     '''
    #     Given simple two vertex indices, alpha and beta, along with the relationship ({2, 3}),
    #     compose R_{lambda} which is the transitive closure for this graph with the edge added
    #     :param alpha: index in self that is the source of relationship np
    #     :param beta: index in self that is the targe of relationship np
    #     :param relationship: r(alpha, beta)
    #     :param compute_closure: if True, compute the closure of R prior to performing the relational composition
    #     :return: transitive closure for Red BLack graph with lambda, new_diameter
    #     '''
    #     pass
    #
    # def rc(self, u, v, color):
    #     '''
    #     rc or Relational Composition is the mechanism used to add a new node
    #     into a Red/Black Graph. (See README.md)
    #
    #     Keyword arguments:
    #     u -- *simple* row vector
    #     v -- *simple* column vector
    #     color -- the coloring of the new node
    #     '''
    #     pass
