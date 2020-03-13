#RedBlackGraph

_redblackgraph is a Numpy Extension Module that makes use
of the Numpy C-API to implement C optimized RedBlack linear algebra
operations on Numpy arrays. 

The primary source file is redblack.c.src which is
essentially a copy of numpy/core/src/multiarray/einsum.c.src with
the sum of products operation being replaced with the avos semiring.

It implements all linear algebra functionality via a modified
einstein summation function.

To update to keep in sync with the numpy implementation, pull in the latest einstein.c.src file
and compare it to redblackgraph.c.src. In addition to the avos semiring, there are 3 other 
modifications:

1) In both the get_combined_dims_view and the get_single_op_view functions use
PyArray_NewFromDescr followed by PyArray_SetBaseObject rather than PyArray_NewFromDescrAndBase (which 
is not in the Numpy C-API)
2) In PyArray_EinsteinSum, rather than PyArray_AssignZero use PyArray_FillWithScalar (also not
in the Numpy C-API)

Also pull in the latest multiarraymodule.c and compare it to redblackgraphmodule.c (only interested 
in the einsteinsum functions)

#RedBlackSparseTools

_sparsetools is a Scipy Extension modules that provides a sparse implementation
of RedBlack linear algebra operations.