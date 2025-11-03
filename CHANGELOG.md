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
- NumPy 2.0 compatibility layer using npy_2_compat.h
- Comprehensive test suite for parity constraints (52 tests)
- Mathematical analysis documenting algebraic structure as two-sorted/ℤ/2ℤ-graded algebra

### Changed
- **BREAKING:** Minimum NumPy version raised to 2.0.0
  - NumPy 1.x is no longer supported
  - Users requiring NumPy 1.x should use redblackgraph v0.5.x
  - Reason: NumPy 2.0 has been stable since June 2024 and provides significant improvements
- Updated C API to use NumPy 2.0 compatible functions:
  - `PyArray_FROM_OF` → `PyArray_FROM_OTF`
  - Direct `descr->elsize` access → `PyDataType_ELSIZE()` macro
  - `NPY_NTYPES` → `NPY_NTYPES_LEGACY` with compatibility layer
- All C extensions updated for NumPy 2.0 opaque structures

### Removed
- **BREAKING:** `redblackgraph.matrix` class (use `redblackgraph.array` instead)
  - `np.matrix` was deprecated in NumPy 1.19 and removed in NumPy 2.0
  - Migration: Replace `rb.matrix(...)` with `rb.array(...)`
- `__config__.py` generation (internal build artifact, no user impact)
- NumPy 1.26 testing from CI (simplified to NumPy 2.x only)
- Deprecated `numpy/noprefix.h` header from C extensions

### Fixed
- Compatibility with NumPy 2.0+ C API changes
- All 167 tests passing on NumPy 2.x across Python 3.10, 3.11, 3.12

## [0.5.0] - Previous Release

(Earlier versions not documented in this changelog)
