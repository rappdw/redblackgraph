# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - TBD

### Added
- **NumPy 2.x support** (backwards compatible with NumPy 1.26+)
- CI testing matrix for both NumPy 1.26 and NumPy 2.1 across Python 3.10, 3.11, 3.12
- Comprehensive CHANGELOG documentation
- NumPy 2.0 compatibility macros for C API

### Removed
- **BREAKING:** `redblackgraph.matrix` class (deprecated since NumPy 1.19)
  - **Migration:** Use `redblackgraph.array` instead
  - Example: Change `rb.matrix([...])` to `rb.array([...])`
- `__config__.py` generation (internal build artifact, no user impact)
- References to deprecated `numpy/noprefix.h` from all C source files

### Changed
- Updated C API to use `PyArray_FROM_OTF` instead of deprecated `PyArray_FROM_OF`
- Dependency constraint: `numpy>=1.26.0,<3.0` (supports both NumPy 1.x and 2.x)
- Replaced `NPY_NTYPES` with `NPY_NTYPES_LEGACY` for NumPy 2.0 compatibility
- Updated `PyArray_Descr` elsize access to use compatibility macro
- Updated documentation to reflect NumPy 1.26+ and 2.x support

### Fixed
- Compatibility with NumPy 1.26.x validated across Python 3.10, 3.11, 3.12
- Compatibility with NumPy 2.1.x validated with C API compatibility layer
- C API deprecation warnings resolved for both NumPy 1.x and 2.x
- All 117 tests passing on both NumPy 1.26 and 2.1

## [0.5.0] - Previous Release

(Earlier versions not documented in this changelog)
