#RedBlackMultiArray

_multiarray is a Numpy Extension Module that makes use
of the Numpy C-API to implement C optimized RedBlack linear algebra
operations on Numpy arrays. 

The primary source file is redblack.c.src which is
essentially a copy of numpy/core/src/multiarray/einsum.c.src with
the sum of products operation being replaced with a non-zero min of
avos operation.

It implements all linear algebra functionality via a modified
einstein summation function.

#RedBlackSparseTools

_sparsetools is a Scipy Extension modules that provides a sparse implementation
of RedBlack linear algebra operations.