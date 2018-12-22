import numpy as np
from numpy import ndarray, asarray
from .avos import einsum
from ._multiarray import warshall, vertex_relational_composition, edge_relational_composition
from redblackgraph.types.transitive_closure import TransitiveClosure

__all__ = ['array', 'matrix']


class _Avos(ndarray):

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
        c_black = int((m + trace) / 2)
        return {
            'red': m - c_black,
            'black': c_black
        }

    def transitive_closure(self) -> TransitiveClosure:
        res = warshall(self)
        return TransitiveClosure(res[0], res[1])

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
        if compute_closure:
            R_star, _ = warshall(self)
        else:
            R_star = self

        # u and v should be rank 1
        if len(u.shape) > 1:
            u = u.reshape((1, R_star.shape[0]))[0]
        if len(v.shape) > 1:
            v = v.reshape((1, R_star.shape[1]))[0]

        out = np.empty(shape=(R_star.shape[0] + 1, R_star.shape[1] + 1), dtype=R_star.dtype).view(type(self))
        return vertex_relational_composition(u, R_star, v, c, out)

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
        if compute_closure:
            # todo: should we determine if the composition increases the diameter, and if so, how would we implement
            R_star, _ = warshall(self)
        else:
            R_star = self

        # loop prevention
        if R_star[beta][alpha] != 0:
            raise ValueError("Relational composition would result in a cycle.")

        return edge_relational_composition(R_star, alpha, beta, relationship)


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
