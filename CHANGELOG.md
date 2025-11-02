# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - TBD

### Added
- **NumPy 2.x support** - Full compatibility with NumPy 2.0+
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
