Red Black Graph - A DAG of Multiple, Interleaved Binary Trees
----------------------------------

# Introduction

Red Black Trees are binary trees such that each node has an extra bit, color (red or black). This color bit us used to balance the tree as modifications are made. In working on data structures to effectively model familial relationships, we find the idea of adding a color bit to a DAG of multiple interleaved binary trees to have utility. The result is a new data structure, operators, and extensions of linear algebra denoted as a "Red Black Graph". For an indepth treatment of the data structure, open the Jupyter notebooks in the notebooks directory. You can easily do this by running:

1. `bin/build-image dev`
2. `bin/run-image notebook`
3. open the url in the log, e.g. http://localhost:8888/?token=1417275a25db329622eeee89e48e28dd6c1bae3edc3eb8d9

This module provides two implementations of Red Blakc Graph. `redblackgraph.simple` has a pure python implementation. It is useful for illustrative purposes. There is an optimized implementation built on scipy's sparse matrix implementation in `redblackgraph.rgm`. 

# Usage Documentation
TODO: see test cases for now, but will be adding documentation... 