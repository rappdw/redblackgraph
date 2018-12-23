Red Black Graph - A DAG of Multiple, Interleaved Binary Trees
----------------------------------

# Introduction

Red Black Graphs are a specific type of graph, a directed acyclic graph of interleaved binary trees.
This data structure resulted from exporation into efficient representations for family history.
This package presents and implements the underlying math as well as discusses some interesting applications.

This python module extends both scipy and numpy and also conforms to [dockerutils](https://github.com/rappdw/docker-utils)
conventions for buildingc and running docker images used in module development. There is a script in the repo bin 
directory that can be used to setup the project for development or to prep for reading the notebook. 
(`bin/setup-project.sh`). You will likely want to create an activate a virtual environment prior to running the script.

# Reading the Notebook

Extensive documentation and examples can be found in the Jupyter notebook, 
"Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb". To access the notebook 
after you've setup the project for development, simply: 
* `run-image notebook`
* `open http://localhost:8888/lab`
 
If you'd prefer to read hard copy, simply run: 

    `bin/generate-pdf.sh notebooks/Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb` 

A pdf file will be generated into the `build/latex-{datestamped}` directory.

# A Note on Implementations

* `redblackgraph.reference` - a pure python implementation. This simple implementation is intended primarily for illustrative purposes.
* `redblackgraph.matrix` and `redblackgrpah.array` - a Numpy C-API extension for efficient computation with the matrix multiplication operator, @, overloaded to support avos sum and product. 
* `redblackgraph.sparse_matrix` - an optimized implementation built on scipy's sparse matrix implementation. 
 