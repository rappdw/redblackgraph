# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - TBD

### Added
- **NumPy 2.x support** - Full compatibility with NumPy 2.0+
- **Explicit constants for red-black algebra** - `RED_ONE` and `BLACK_ONE`
  - `RED_ONE` (-1): Identity for even values, annihilator for odd values
  - `BLACK_ONE` (1): Identity for odd values, annihilator for even values  
  - Helper functions: `red_one_for_dtype()`, `black_one_for_dtype()`, `is_red_one()`, `is_black_one()`
  - Makes parity-based algebraic semantics explicit and self-documenting
- **Parity identity constraints in AVOS product** - Asymmetric identity behavior
  - LEFT identity: Acts as starting point marker (no filtering)
  - RIGHT identity: Acts as gender/parity filter (enforces parity constraints)
  - Prevents impossible relationships like "father's female self" or "mother's male self"
  - Implemented across all backends: Python reference, C core, C++ sparse, Cython
  - See notebook for detailed mathematical justification and examples
- CI testing matrix for NumPy 1.26 and 2.1 across Python 3.10, 3.11, 3.12
- NumPy 2.0 compatibility layer using npy_2_compat.h

### Removed
- **BREAKING:** `redblackgraph.matrix` class (use `redblackgraph.array` instead)
  - `np.matrix` was deprecated in NumPy 1.19 and removed in NumPy 2.0
  - Migration: Replace `rb.matrix(...)` with `rb.array(...)`
- `__config__.py` generation (internal build artifact, no user impact)
- Deprecated `numpy/noprefix.h` header from C extensions

### Changed
- Updated C API to use NumPy 2.0 compatible functions:
  - `PyArray_FROM_OF` → `PyArray_FROM_OTF`
  - Direct `descr->elsize` access → `PyDataType_ELSIZE()` macro
  - `NPY_NTYPES` → `NPY_NTYPES_LEGACY` with compatibility layer
- Dependency constraint: `numpy>=1.26.0,<3.0` (supports NumPy 1.x and 2.x)
- All C extensions updated for NumPy 2.0 opaque structures

### Fixed
- Compatibility with NumPy 2.0+ C API changes
- All 117 tests passing on NumPy 1.26.x and 2.1.x

## [0.5.0] - Previous Release

(Earlier versions not documented in this changelog)
