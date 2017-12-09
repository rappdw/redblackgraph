import numpy as np
from numpy import ndarray, asarray
from redblackgraph.core.einsumfunc import einsum

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
        raise ValueError(f"Unexpected dimensionality. self: {self.dim}, other: {other.dim}")

    def __rmatmul__(self, other):
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
        raise ValueError(f"Unexpected dimensionality. self: {self.dim}, other: {other.dim}")

    def __imatmul__(self, other):
        if self.ndim == 1:
            if other.ndim == 1:
                return einsum('i,i', self, other, self, avos=True)
            if other.ndim == 2:
                return einsum('i,...ij', self, other, self, avos=True)
        if self.ndim == 2:
            if other.ndim == 1:
                return einsum('...i,i', self, other, self, avos=True)
            if other.ndim == 2:
                return einsum('ij,jk', self, other, self, avos=True)
        raise ValueError(f"Unexpected dimensionality. self: {self.dim}, other: {other.dim}")


class array(_Avos, ndarray):
    def __new__(cls, *args, **kwargs):
        return asarray(*args, **kwargs).view(cls)

    def __matmul__(self, other):
        return super(array, self).__matmul__(other).view(array)

    def __rmatmul__(self, other):
        return super(array, self).__rmatmul__(other).view(array)

    def __imatmul__(self, other):
        return super(array, self).__imatmul__(other).view(array)


class matrix(_Avos, np.matrix):
    def __new__(cls, data, dtype=None, copy=True):
        return super(matrix, cls).__new__(cls, data, dtype=dtype, copy=copy)

    def __matmul__(self, other):
        return super(matrix, self).__matmul__(other).view(matrix)

    def __rmatmul__(self, other):
        return super(matrix, self).__rmatmul__(other).view(matrix)

    def __imatmul__(self, other):
        return super(matrix, self).__imatmul__(other).view(matrix)
