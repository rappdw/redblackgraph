Red Black Graph - A DAG of Multiple, Interleaved Binary Trees
----------------------------------

# Introduction

Red Black Trees are binary trees such that each node has an extra bit, color (red or black). This color bit us used to balance the tree as modifications are made. In working on data structures to effectively model familial relationships, we find the idea of adding a color bit to a DAG of multiple interleaved binary trees to have utility. The result is a new data structure, operators, and extensions of linear algebra denoted as a "Red Black Graph". For an indepth treatment of the data structure, open the Jupyter notebooks in the notebooks directory. You can easily do this by running:

1. `pip3 install dockerutils`
2. `build-image notebook`
3. `run-image notebook`
4. open the url in the log, e.g. http://localhost:8888/?token=1417275a25db329622eeee89e48e28dd6c1bae3edc3eb8d9
5. Navigate to the notebooks sub-dir and explore detailed notebooks from there including:
    1. *Red Black Graphs* - An introduction to the data structure
    2. *Linear Algebra of Red Black Graphs* - A more in depth treatment of some of the linear algebra properites of a Red Black Graph
    3. *Python Implementation* - A discussion of some of the Python implementation approaches

This module provides serveral implementations of Red Black Graph. 
* `redblackgraph.simple` - a pure python implementation. This simple implementation is intended for illustrative purposes only.
* `redblackgraph.matrix` and `redblackgrpah.array` - a Numpy C-API extension for efficient computation with the matrix multiplication operator, @, overloaded to support Red Black Graph linear algebra. 
* `redblackgraph.sparse_matrix` - an optimized implementation built on scipy's sparse matrix implementation. 

**Note:** conforms to and utilizes [dockerutils](https://github.com/rappdw/docker-utils) conventions. 