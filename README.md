[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rappdw/f559859044b3e491a5dd6d75887c5145/raw/redblackgraph-coverage.json)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![PyPi](https://img.shields.io/pypi/v/redblackgraph.svg)](https://pypi.org/project/redblackgraph/) 
[![PyPi](https://img.shields.io/pypi/wheel/redblackgraph.svg)](https://pypi.org/project/redblackgraph/) 
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) 
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) 

Red-Black Graph - A DAG of Multiple, Interleaved Binary Trees
----------------------------------

# Introduction

Red-Black Graphs are a specific type of graph, a directed acyclic graph of interleaved binary trees.
This data structure resulted from exploration of efficient representations for family history.
This package presents and implements the underlying linear algebra as well as discusses some interesting applications.

This python module extends both scipy and numpy and also conforms to [dockerutils](https://github.com/rappdw/docker-utils)
conventions for building and running docker images used in module development. There is a script in the bin 
directory that can be used to setup the project for development or to prep for reading the notebook. 
(`bin/setup-project.sh`). You will want to create an activate a virtual environment prior to running the script.

# Reading the Notebook

A research paper describing the linear algebra underlying Red-Black graphs as well as examples of application can be found in the Jupyter notebook, 
"Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb". To access the notebook 
after you've setup the project for development, simply: 
* `run-image notebook`
* `open http://localhost:8888/lab`
 
If you'd prefer to read hard copy, simply run: 

    `bin/generate-pdf.sh notebooks/Red Black Graph - A DAG of Multiple, Interleaved Binary Trees.ipynb` 

A pdf file will be generated into the `build/latex-{datestamped}` directory.

# To Try Things Out...

Run the following:

```shell script
# use crawl-fs to extract a sample data set from FamilySearch
pip install fs-crawler
crawl-fs -i <FamilySearch Ids to seed crawl> -o <output-directory> -b <name portion of output file>

# this will generate a <name>.vertices.csv and <name>.edges.csv file which can be ingested into a RedBlackGraph
pip install RedBlackGraph
# use rbgcf to generate both a simple form and cannonical form of a Red Black Graph (xlsx files)
rbgcf -f <directory and base name of vertices and edges file> -o <output-directory>

# Use excel to view output
 
```

# Building from Source

RedBlackGraph uses the Meson build system (as of version 0.5.0, migrated from numpy.distutils).

## Requirements
- Python 3.10, 3.11, or 3.12
- Meson >= 1.2.0
- Ninja build tool
- Cython >= 3.0
- NumPy >= 1.26

## Build and Install
```bash
# Install build dependencies
pip install meson-python meson ninja cython numpy

# Build and install in development mode
pip install -e .

# Or build wheel
pip install build
python -m build
```

The Meson build system compiles all C/C++ extensions and Cython modules automatically.

# Building and Publishing Wheels

RedBlackGraph uses `cibuildwheel` to build wheels for multiple platforms and Python versions.

## Quick Start

```bash
# Build wheels for current platform
./bin/build-wheels-cibuildwheel.sh

# Or use cibuildwheel directly
pip install cibuildwheel
cibuildwheel --platform auto --output-dir wheelhouse
```

## Automated Release

Wheels are automatically built and published to PyPI when a version tag is pushed:

```bash
git tag -a v0.5.1 -m "Release version 0.5.1"
git push origin v0.5.1
```

For detailed instructions, see:
- **[PyPI Publishing Guide](docs/PYPI_PUBLISHING.md)** - Complete guide for building and publishing
- **[Release Checklist](RELEASE_CHECKLIST.md)** - Quick reference for releases

# A Note on Implementations

* `redblackgraph.reference` - a pure python implementation. This simple implementation is intended primarily for illustrative purposes.
* `redblackgraph.matrix` and `redblackgrpah.array` - a Numpy C-API extension for efficient computation with the matrix multiplication operator, @, overloaded to support avos sum and product. 
* `redblackgraph.sparse_matrix` - an optimized implementation built on scipy's sparse matrix implementation. 
 
