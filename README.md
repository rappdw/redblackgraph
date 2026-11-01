[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rappdw/redblackgraph)
[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rappdw/f559859044b3e491a5dd6d75887c5145/raw/redblackgraph-coverage.json)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![PyPi](https://img.shields.io/pypi/v/redblackgraph.svg)](https://pypi.org/project/redblackgraph/)
[![PyPi](https://img.shields.io/pypi/wheel/redblackgraph.svg)](https://pypi.org/project/redblackgraph/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

# Red Black Graph

A Red Black Graph is a directed acyclic graph of multiple, interleaved binary trees. Each vertex is
colored red or black, and each vertex has at most one outgoing edge to a vertex of each color. The
structure came out of looking for an efficient representation of family history: vertices are
people, red is male, black is female, and edges point from a child to its parents.

This package implements the underlying algebra, the *avos* semiring, together with graph algorithms
built on it: transitive closure, relationship calculation, cycle detection and prevention,
incremental updates, component identification and canonical ordering. It provides dense (NumPy),
sparse (SciPy) and GPU (CuPy) implementations.

## Installation

```bash
pip install redblackgraph

# optional extras
pip install "redblackgraph[io]"    # CSV/Excel readers and writers (fs-crawler, XlsxWriter)
pip install "redblackgraph[gpu]"   # GPU support (CuPy for CUDA 12)
```

Wheels are published for Python 3.10–3.12 on Linux, macOS and Windows. NumPy 2.x is required (use
v0.5.x for NumPy 1.x).

## Quick Example

Relationships are encoded as Ahnentafel (pedigree chart) numbers: 2 is a father, 3 a mother, 4 a
paternal grandfather, and in general the father of `n` is `2n` and the mother is `2n+1`. The
diagonal records each person's color: `RED_ONE` (-1) for male and `BLACK_ONE` (1) for female.

```python
import redblackgraph as rb
from redblackgraph import RED_ONE, BLACK_ONE
from redblackgraph.reference import calculate_relationship

# 0: daughter, 1: son, 2: father, 3: mother, 4: paternal grandfather
A = rb.array([[BLACK_ONE, 0, 2, 3, 0],
              [0, RED_ONE, 2, 3, 0],
              [0, 0, RED_ONE, 0, 2],
              [0, 0, 0, BLACK_ONE, 0],
              [0, 0, 0, 0, RED_ONE]])

closure = A.transitive_closure()
closure.W[0, 4]    # 4: the daughter's paternal grandfather
closure.diameter   # 2: the longest chain of generations
calculate_relationship(closure.W[0], closure.W[1])
# Relationship(common_ancestor=2, relationship='sibling')
```

## The Paper and the Mathematics

The notebook
[`notebooks/red-black-graphs.ipynb`](notebooks/red-black-graphs.ipynb), *Red Black Graphs and the Avos
Semiring*, is a research paper describing Red Black Graphs, the avos algebra and its applications,
with runnable examples. To read it locally:

```bash
pip install jupyterlab
jupyter lab notebooks/red-black-graphs.ipynb
```

To produce a PDF, run `bin/generate-pdf.sh notebooks/red-black-graphs.ipynb`. It needs `nbconvert`, `pandoc` and
`xelatex` (for example from TeX Live), and writes the PDF to `build/latex-<timestamp>/`.

Related documents:

- [Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md): the formal development with full
  proofs. The avos sum and product form a semiring (the shortlex path semiring), which is what makes
  the closure algorithms and incremental updates correct.
- [Applications Beyond Genealogy](docs/APPLICATIONS_BEYOND_GENEALOGY.md): other domains where the
  structure applies.

## Working with Family Data

To try the algorithms on a realistic graph without any external data, synthesize one:

```bash
rbg-synthesize -g 6 -n 50 -s 1 -o family    # writes family.vertices.csv and family.edges.csv
```

To use real data, [fs-crawler](https://github.com/rappdw/fs-crawler) can extract a data set from
FamilySearch in the same format:

```bash
pip install fs-crawler
crawl-fs -i <FamilySearch ids to seed the crawl> -o <output-directory> -b <base name>
```

From a source checkout, `scripts/rbg` reads these files, computes the closure and writes the results
to Excel (it needs the `io` extra, and the output directory must already exist):

```bash
mkdir -p out
python scripts/rbg -f family -o out run -c    # canonical ordering -> out/family.4.canonical.xlsx
```

Run `python scripts/rbg run --help` for the other analyses (closure, topological ordering, row and
column statistics).

## Implementations

All implementations share the same avos semantics.

| Module | Description |
| --- | --- |
| `redblackgraph.reference` | Pure Python. Simple implementations intended for illustration and as a test oracle. |
| `redblackgraph.core` (`rb.array`) | NumPy C extension. `@` is overloaded as the avos matrix product. Best for small, dense graphs. |
| `redblackgraph.sparse` (`rb.rb_matrix`) | SciPy sparse matrices with Cython graph algorithms. Best for genealogical graphs, which are typically far below 1% dense. Includes an O(V + E + nnz) closure for DAGs. |
| `redblackgraph.gpu` | CuPy with inline CUDA kernels: sparse matrix multiply (SpGEMM) and transitive closure on the GPU. |

## GPU Acceleration

The GPU module (`redblackgraph.gpu`) provides two transitive closure algorithms:

1. **Repeated squaring** (`transitive_closure_gpu`) works for any graph. It computes
   TC(A) = A + A² + A⁴ + ... with CUDA SpGEMM, keeping all data on the GPU between iterations.
2. **Level-parallel DAG propagation** (`transitive_closure_dag_gpu`) is specialized for DAGs
   (triangular matrices). It processes vertices by topological level, in parallel within each level.

### Performance

Benchmarked on synthesized family DAGs (lower-triangular adjacency matrices). The CPU column uses
the Cython O(V + E + nnz) topological propagation algorithm.

| Vertices | NNZ | CPU-DAG (s) | GPU-Sqr (s) | GPU-DAG (s) | Best GPU/CPU |
|----------|-----|-------------|-------------|-------------|-------------|
| 442 | 1,226 | 0.0020 | 0.0039 | 0.0032 | 1.6x CPU |
| 1,326 | 3,728 | 0.0080 | 0.0055 | 0.0038 | **2.1x GPU** |
| 2,144 | 5,932 | 0.0129 | 0.0059 | 0.0043 | **3.0x GPU** |
| 4,701 | 13,103 | 0.0292 | 0.0073 | 0.0057 | **5.1x GPU** |
| 11,012 | 30,536 | 0.0700 | 0.0125 | 0.0087 | **8.1x GPU** |
| 21,162 | 58,486 | 0.1380 | 0.0268 | 0.0138 | **10.0x GPU** |

The GPU overtakes the CPU at about 1,000 vertices, and the DAG kernel nearly doubles the speedup of
repeated squaring at scale. Run `python bench_closure.py` to reproduce the benchmark on your
hardware.

### Usage

```bash
pip install "redblackgraph[gpu]"   # CUDA 12; for CUDA 11 install cupy-cuda11x instead
```

GPU features are optional. Everything else works without CuPy.

```python
from redblackgraph.gpu import CSRMatrixGPU, transitive_closure_dag_gpu

A_gpu = CSRMatrixGPU.from_cpu(my_sparse_matrix)     # copy a scipy CSR matrix to the GPU
closure, diameter = transitive_closure_dag_gpu(A_gpu)
result = closure.to_cpu()                           # copy back as a scipy CSR matrix
```

## Building from Source

RedBlackGraph uses the [Meson](https://mesonbuild.com) build system, with `meson-python` as the
build backend.

Requirements:

- Python 3.10, 3.11 or 3.12
- A C/C++ compiler and the Ninja build tool (e.g. `sudo apt install ninja-build`)
- Build dependencies: `meson>=1.2.0`, `meson-python`, `cython>=3.0`, `tempita`, `numpy>=2.0`

Editable installs must use `--no-build-isolation`, because the editable loader needs a persistent
build directory.

### With uv

```bash
git clone https://github.com/rappdw/redblackgraph.git
cd redblackgraph
./bin/setup-uv.sh           # CPU only
./bin/setup-uv.sh --gpu     # include CuPy
source .venv/bin/activate
```

The script creates `.venv`, installs the build and test dependencies, initializes the `fs-crawler`
submodule if needed, and performs the editable install.

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install meson-python meson ninja cython tempita numpy
pip install -e ".[test,io]" --no-build-isolation
```

Re-run the last command after changing C, C++ or Cython sources. To build a wheel instead:
`pip install build && python -m build`.

## Testing

```bash
pytest tests                                            # all tests
pytest --cov=redblackgraph --cov-fail-under=65 tests    # with coverage, as in CI
pylint redblackgraph --rcfile=.pylintrc -d C,R          # lint
```

GPU tests run only when CuPy and an NVIDIA GPU are available, and are skipped otherwise.
`tests/avos/test_algebra_properties.py` checks the theorems in the
[Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md) against every implementation.

## Releases

Wheels are built with `cibuildwheel` and published to PyPI automatically when a version tag is
pushed:

```bash
git tag -a v0.7.0 -m "Release version 0.7.0"
git push origin v0.7.0
```

To build wheels locally, run `./bin/build-wheels-cibuildwheel.sh`. See the
[PyPI Publishing Guide](docs/PYPI_PUBLISHING.md) and the [Release Checklist](RELEASE_CHECKLIST.md)
for details.

## License

[AGPL-3.0-or-later](LICENSE)
