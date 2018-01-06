Red Black Graph - A DAG of Multiple, Interleaved Binary Trees
----------------------------------

# Introduction

Red Black Graphs are a specific type of graph used to model things like familial relationships.
This package presents (in a Jupyter notebook) and implements the underlying math as well as
discusses some interesting applications.

As this python module is a numpy extension, you must install numpy before running setup.py or
trying to pip install the module. There is a script in the repo bin directory that can be used
to setup the project for development or to prep for reading the notebook. (`bin/setup-project.sh`)

# Reading the Notebook

Extensive documentation and examples can be found in the Jupyter notebook, 
"Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb". To access the notebook 
after you've setup the project for development, simply: `run-image notebook`, observe the URL 
for the Jupyter server, open the URL in a browser and navigate to the notebooks directory to
open the notebook. If you'd prefer to read hard copy, simply run 
`bin/generate-pdf.sh notebooks/Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb`. 
A pdf file will be generated into the `build/latex-{datestamped}` directory.

# A Note on Implementations

* `redblackgraph.simple` - a pure python implementation. This simple implementation is intended for illustrative purposes only.
* `redblackgraph.matrix` and `redblackgrpah.array` - a Numpy C-API extension for efficient computation with the matrix multiplication operator, @, overloaded to support avos matrix products. 
* `redblackgraph.sparse_matrix` - an optimized implementation built on scipy's sparse matrix implementation. 

**Note:** This repository conforms to and utilizes [dockerutils](https://github.com/rappdw/docker-utils) conventions. 