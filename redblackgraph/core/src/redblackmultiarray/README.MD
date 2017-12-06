RedBlackMultiArray
------------------

RedBlackMultiArray is a Python Extension Module that makes use
of the Numpy C-API to implement C optimized RedBlack linear algebra
operations on Numpy arrays. 

The primary source file is redblack.c.src which is
essentially a copy of numpy/core/src/multiarray/einsum.c.src with
the sum of products operation being replaced with a non-zero min of
avos operation.

Additionally, just enough source from multiarraymodule.c has been lifted
to initialize the module and provide the entry point that parses the 
args to einsum and dispatches execution to the main entry point in
redblack.c.src (PyArray_EinsteinSum2).

H1 - Building the RedBlackMultiArray extension module

* utilize conv_template.py from numpy/disutils to process redblack.c.src