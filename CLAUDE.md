# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RedBlackGraph is a Python/C/Cython/CUDA library implementing AVOS (Algebraic Vertex-Ordered Semiring) algebra for computing directed acyclic graphs as interleaved binary trees. Primary application is genealogical relationship modeling.

## Build & Development

**Build system**: Meson (not setuptools). Build backend is `meson-python`.

```bash
# Install in editable mode (required: --no-build-isolation)
pip install -e ".[test]" --no-build-isolation

# Rebuild after C/Cython changes
pip install -e . --no-build-isolation
```

Build dependencies: meson>=1.2.0, ninja, cython>=3.0, tempita>=0.5.2, numpy>=2.0.

## Testing

```bash
# Run all tests
pytest tests

# Run a single test file
pytest tests/gpu/test_spgemm.py

# Run a single test
pytest tests/gpu/test_spgemm.py::TestSpGEMM::test_name

# Run with coverage (CI uses 65% minimum)
pytest --cov=redblackgraph --cov-fail-under=65 tests
```

GPU tests require CuPy and an NVIDIA GPU. They are skipped automatically if unavailable.

## Linting

```bash
pylint redblackgraph --rcfile=.pylintrc -d C,R
```

## Architecture

Three computation backends share the same AVOS algebra semantics:

1. **Dense (NumPy C extension)** — `redblackgraph/core/` — C code generated from Tempita templates (`.c.in`, `.h.in`) and NumPy `.c.src` multi-type generators. Provides `array` class with `@` operator for AVOS matrix multiply.

2. **Sparse (SciPy + Cython)** — `redblackgraph/sparse/` — `rb_matrix` subclasses `scipy.sparse.csr_matrix`. Graph algorithms in `sparse/csgraph/` are Cython extensions (`.pyx`): transitive closure, topological sort, canonical ordering, relational composition.

3. **GPU (CuPy + CUDA)** — `redblackgraph/gpu/` — Pure Python using CuPy's `RawKernel` for inline CUDA. Two-phase SpGEMM: symbolic phase computes output sparsity pattern via global memory hash tables, numeric phase computes AVOS values with `atomicMin`.

### AVOS Algebra Essentials

- **Set**: integers with special values `RED_ONE = -1` and `BLACK_ONE = 1` (defined in `constants.py`)
- **Addition (⊕)**: `min(x, y)` treating 0 as infinity
- **Multiplication (⊗)**: bit-shift composition with parity constraints
- RED_ONE/BLACK_ONE have asymmetric identity/annihilator behavior based on parity — this is NOT a classical semiring
- All three backends must maintain identical parity constraint semantics

### GPU backend details

The GPU module (`redblackgraph/gpu/`) is pure Python with inline CUDA via CuPy `RawKernel`:

- **`CSRMatrixGPU`** — Sparse matrix with raw int32 buffers (not CuPy sparse). Supports `@` operator, `copy()`, `eliminate_zeros()`, `transitive_closure()`, `prefetch()` (for Grace Hopper UVM).
- **`spgemm(A, B=None)`** — Two-phase SpGEMM: symbolic phase computes sparsity pattern via per-row hash tables in global memory, numeric phase computes AVOS values using `atomicMin`. Supports both `A @ A` (self-multiply with triangular optimization) and general `A @ B`.
- **`transitive_closure_gpu(A)`** — Repeated squaring on GPU: `TC(A) = A + A² + A⁴ + ...`. All data stays GPU-resident between iterations (no CPU round-trips).
- **Kernel singletons**: `SymbolicPhase`, `NumericPhase`, and `AVOSKernels` classes compile CUDA kernels once via `cp.RawKernel` and cache via `get_*()` functions.

### Key code generation pipeline (dense backend)

Tempita templates (`.h.in`, `.c.in`) → processed by `tempita` → C source → compiled via Meson. NumPy `.c.src` files use `/**begin repeat*/` blocks for multi-type generation. Both happen at build time.

## Version Management

Uses `setuptools_scm` — version derived from git tags. Release tags follow `v*.*.*` pattern.
