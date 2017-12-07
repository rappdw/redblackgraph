from numpy import ndarray, asarray, matrix
from redblackgraph.core.einsumfunc import einsum

__all__ = ['rbarray', 'rbmatrix']


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


class rbarray(_Avos, ndarray):
    def __new__(cls, *args, **kwargs):
        return asarray(*args, **kwargs).view(cls)

    def __matmul__(self, other):
        return super(rbarray, self).__matmul__(other).view(rbarray)

    def __rmatmul__(self, other):
        return super(rbarray, self).__rmatmul__(other).view(rbarray)

    def __imatmul__(self, other):
        return super(rbarray, self).__imatmul__(other).view(rbarray)


class rbmatrix(_Avos, matrix):
    def __new__(cls, data, dtype=None, copy=True):
        return super(rbmatrix, cls).__new__(cls, data, dtype=dtype, copy=copy)

    def __matmul__(self, other):
        return super(rbmatrix, self).__matmul__(other).view(rbmatrix)

    def __rmatmul__(self, other):
        return super(rbmatrix, self).__rmatmul__(other).view(rbmatrix)

    def __imatmul__(self, other):
        return super(rbmatrix, self).__imatmul__(other).view(rbmatrix)
