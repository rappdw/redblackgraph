import numpy as np
from numpy import ndarray, asarray
from . import einsum, warshall, relational_composition

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

    def relational_composition(self, u, v, compute_closure=False):
        '''
        Given simple row vector u, and simple column vector v where
        u, v represent a vertex, lambda, not currently represented in self, compose A_{\lambda}
        which is the transitive closure for this graph with lambda included
        :param u: simple row vector for new vertex, lambda
        :param v: simple column vector for new vertex, lambda
        :param c: color of the new vertex, either -1 or 1
        :param compute_closure: if True, compute the closure of A prior to performing the relational composition
        :return: transitive closure for Red BLack graph with lambda, new_diameter
        '''
        if compute_closure:
            # todo: should we determine if the composition increases the diameter, and if so, how would we implement
            A_star, _ = warshall(self)
        else:
            A_star = self

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
        M = A_star.shape[0]
        Nu = u.shape
        Nv = v.shape
        assert Nu[0] == 1
        assert Nv[1] == 1
        assert Nu[1] == Nv[0]
        assert Nu[1] == M + 1
        assert u[0][-1] == v[-1][0]

        # see discussion: https://stackoverflow.com/questions/26285595/generalized-universal-function-in-numpy
        # unfortunately, his proposal was never merged into Numpy, so the wrapper approach seems to be
        # best way to do this

        # rather than using a wrapper, go a head and in the extra value in the u and v arrays and then
        # let that determine the output size

        uc_lambda = u[:,:-1] @ A_star
        vc_lambda = A_star @ v[:-1,:]
        # add the last element from u,v into uc_lambda and vc_lambda and
        # collapse these down to rank 1 arrays, as that is what gufunc is expecting
        uc_lambda = np.append(uc_lambda[0], u[0][-1]).view(type(u))
        vc_lambda = np.append(vc_lambda.reshape(1, vc_lambda.shape[0])[0], v[-1][0]).view(type(v))

        return relational_composition(uc_lambda, A_star, vc_lambda)

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
