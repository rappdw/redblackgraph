import numpy as np
from numpy import ndarray, asarray
from . import einsum, warshall, vertex_relational_composition, vertex_relational_composition2, edge_relational_composition

__all__ = ['array', 'matrix']


class _Avos():

    def __matmul__(self, other):
        if self.ndim == 1:
            if other.ndim == 1:
                return einsum('i,i', self, other, avos=True)
            if other.ndim == 2:
                return einsum('i,...ij', self, other, avos=True)
        if self.ndim == 2:
            if other.ndim == 1:
                return einsum('...i,i', self, other, avos=True)
            if other.ndim == 2:
                return einsum('ij,jk', self, other, avos=True)
        raise ValueError(f"Unexpected dimensionality. self: {self.ndim}, other: {other.ndim}")

    def __rmatmul__(self, other):
        if self.ndim == 1:
            if other.ndim == 1:
                return einsum('i,i', other, self, avos=True)
            if other.ndim == 2:
                return einsum('...i,i', other, self, avos=True)
        if self.ndim == 2:
            if other.ndim == 1:
                return einsum('i,...ij', other, self, avos=True)
            if other.ndim == 2:
                return einsum('ij,jk', other, self, avos=True)
        raise ValueError(f"Unexpected dimensionality. self: {self.ndim}, other: {other.ndim}")

    def cardinality(self):
        trace = np.trace(self)
        m = self.shape[0]
        c_black = int((self.shape[0] + trace) / 2)
        return {
            'red': self.shape[0] - c_black,
            'black': c_black
        }

    def transitive_closure(self):
        return warshall(self)

    def vertex_relational_composition(self, u, v, c, compute_closure=False):
        '''
        Given simple row vector u, and simple column vector v where
        u, v represent a vertex, lambda, not currently represented in self, compose R_{\lambda}
        which is the transitive closure for this graph with lambda included
        :param u: simple row vector for new vertex, lambda
        :param v: simple column vector for new vertex, lambda
        :param c: color of the new vertex, either -1 or 1
        :param compute_closure: if True, compute the closure of R prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda
        '''
        if compute_closure:
            # todo: should we determine if the composition increases the diameter, and if so, how would we implement
            R_star, _ = warshall(self)
        else:
            R_star = self

        # if u/v are rank 1 arrays, reshape them
        if len(u.shape) == 1:
            u = u.reshape(1, u.shape[0])
        if len(v.shape) == 1:
            v = v.reshape(v.shape[0], 1)

        # discussion: this perhaps should all be implemented in a c-extension, either an API or a gufunc.
        # however, the overhead of making the calls required to set this up in python is minimal compared
        # to the linalg execution. Plus the code is much more readable and understandable in python.

        # validate the constraints, e.g. u/v are row/column vectors sized 1 more than this matrix
        # the last dimension of u/v is the same, u/v are not rank 1 arrays
        M = R_star.shape[0]
        Nu = u.shape
        Nv = v.shape
        assert Nu[0] == 1
        assert Nv[1] == 1
        assert Nu[1] == Nv[0]
        assert Nu[1] == M

        # see discussion: https://stackoverflow.com/questions/26285595/generalized-universal-function-in-numpy
        # unfortunately, his proposal was never merged into Numpy, so the wrapper approach seems to be
        # best way to do this

        # rather than using a wrapper, go a head and in the extra value in the u and v arrays and then
        # let that determine the output size

        uc_lambda = u @ R_star
        vc_lambda = R_star @ v

        # add the last element from u,v into uc_lambda and vc_lambda and
        # collapse these down to rank 1 arrays, as that is what gufunc is expecting
        uc_lambda = np.append(uc_lambda[0], c).view(type(u))
        vc_lambda = np.append(vc_lambda.reshape(1, vc_lambda.shape[0])[0], c).view(type(v))

        return vertex_relational_composition(uc_lambda, R_star, vc_lambda)

    def vertex_relational_composition2(self, u, v, c, compute_closure=False):
        '''
        Given simple row vector u, and simple column vector v where
        u, v represent a vertex, lambda, not currently represented in self, compose R_{\lambda}
        which is the transitive closure for this graph with lambda included
        :param u: simple row vector for new vertex, lambda
        :param v: simple column vector for new vertex, lambda
        :param c: color of the new vertex, either -1 or 1
        :param compute_closure: if True, compute the closure of R prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda
        '''
        if compute_closure:
            R_star, _ = warshall(self)
        else:
            R_star = self
        out = np.empty(shape=(R_star.shape[0]+1, R_star.shape[1]+1), dtype=R_star.dtype)
        return vertex_relational_composition2(u, R_star, v, c, out)

    def edge_relational_composition(self, alpha, beta, np, compute_closure=False):
        '''
        Given simple two vertex indices, alpha and beta, along with the relationship ({2, 3}),
        compose R_{\lambda} which is the transitive closure for this graph with the edge added
        :param alpha: index in self that is the source of relationship np
        :param beta: index in self that is the targe of relationship np
        :param np: the pedigree number of the relationship from alpha to beta
        :param compute_closure: if True, compute the closure of R prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda, new_diameter
        '''
        if compute_closure:
            # todo: should we determine if the composition increases the diameter, and if so, how would we implement
            R_star, _ = warshall(self)
        else:
            R_star = self

        # loop prevention
        if R_star[beta][alpha] != 0:
            raise ValueError("Relational composition would result in a cycle.")

        return edge_relational_composition(R_star, alpha, beta, np)


class array(_Avos, ndarray):
    def __new__(cls, *args, **kwargs):
        return asarray(*args, **kwargs).view(cls)

    def __matmul__(self, other):
        return super(array, self).__matmul__(other).view(array)

    def __rmatmul__(self, other):
        return super(array, self).__rmatmul__(other).view(array)


class matrix(_Avos, np.matrix):
    def __new__(cls, data, dtype=None, copy=True):
        return super(matrix, cls).__new__(cls, data, dtype=dtype, copy=copy)

    def __matmul__(self, other):
        return super(matrix, self).__matmul__(other).view(matrix)

    def __rmatmul__(self, other):
        return super(matrix, self).__rmatmul__(other).view(matrix)
