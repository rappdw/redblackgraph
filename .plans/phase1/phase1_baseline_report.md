# Phase 1 Baseline Report

**Date**: 2025-10-21  
**Purpose**: Document current state before Phase 1 migration  
**Branch**: master  
**Sprint**: Phase 1, Sprint 1

---

## Executive Summary

This document captures the baseline state of RedBlackGraph before Phase 1 migration begins. The current codebase (master branch) is functional with Python 3.6-3.8 but **cannot build with modern NumPy (1.23+)** due to deprecated `numpy.distutils` usage.

### Key Findings

- **Current Python Support**: 3.6, 3.7, 3.8, 3.9, 3.10 (per classifiers)
- **Actual Tested Versions**: 3.6, 3.7, 3.8 (per Travis CI)
- **Build System**: numpy.distutils (deprecated since NumPy 1.23.0)
- **Critical Issue**: Cannot build with NumPy >= 1.23 or Python 3.10+
- **Supported Platforms**: Linux, macOS
- **Test Framework**: pytest via `bin/test -u`
- **Test Duration**: < 5 minutes per Python version

---

## Current State Assessment

### Build System Analysis

**Status**: ⚠️ **BLOCKED** - Cannot build with modern dependencies

The project uses `numpy.distutils` which was deprecated in NumPy 1.23.0 (2022) and will be removed in NumPy 2.0.

#### Attempt to Build on Python 3.8

```bash
# Command
uv pip install --python .venv-baseline-3.8 -e ".[dev,test]"

# Result: FAILED
# Error: ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

**Root Cause**: Modern NumPy (>= 1.23) has deprecated numpy.distutils, and Python 3.12+ has removed distutils entirely from stdlib.

#### Attempt to Build on Python 3.10

```bash
# Command
uv pip install --python .venv-3.10 -e ".[dev,test]"

# Result: FAILED
# Error: NPY_NO_EXPORT macro not found in NumPy C API
```

**Root Cause**: C extensions use `NPY_NO_EXPORT` macro which doesn't exist in modern NumPy.

---

## Dependency Analysis

### Current Dependencies (from setup.py)

```python
install_requires=[
    'dataclasses;python_version<"3.7"',  # Backport for old Python
    'numpy>=0.14.0',                      # Very old minimum
    'scipy',                              # No version constraint
    'XlsxWriter',                         # No version constraint
    'fs-crawler>=0.3.2'                   # External dependency
]

setup_requires=[
    'numpy>=0.18.1',                      # Very old minimum
    'cython'                              # No version constraint
]
```

### Issues Identified

1. **numpy>=0.14.0**: Too old (NumPy 0.14 is from ~2015)
2. **No upper bounds**: Can install incompatible NumPy 2.0
3. **dataclasses backport**: Needed for Python < 3.7
4. **scipy**: No version constraints
5. **cython**: No version constraints

### Target Dependencies (Phase 1)

```python
install_requires=[
    'numpy>=1.26.0,<2.0',
    'scipy>=1.11.0',
    'XlsxWriter',
    'fs-crawler>=0.3.2'
]

setup_requires=[
    'numpy>=1.26.0,<2.0',
    'cython>=3.0'
]

python_requires='>=3.10'
```

---

## Code Structure

### Package Hierarchy

```
redblackgraph/
├── core/              # C extensions (4 .c.src files)
├── sparse/            # Sparse matrix implementations
│   └── csgraph/       # Cython extensions (7 .pyx files)
├── reference/         # Pure Python reference implementations
├── types/             # Type definitions
└── util/              # Utility functions
```

### Setup Files (5 total)

All setup files use `numpy.distutils`:

1. **./setup.py** - Root configuration
   - Uses `versioneer` for version management
   - Configures build_ext with GNU linker tweaks
   - Generates Cython code via `tools/cythonize.py`

2. **./redblackgraph/setup.py** - Package hierarchy
   - Adds subpackages: core, sparse, reference, types, util

3. **./redblackgraph/core/setup.py** - C extensions
   - Builds 4 C extension modules from .c.src templates
   - Uses numpy.distutils template processing

4. **./redblackgraph/sparse/setup.py** - Sparse extensions
   - Configuration for sparse matrix extensions

5. **./redblackgraph/sparse/csgraph/setup.py** - Cython extensions
   - Builds 7 Cython modules

### C Extensions (Template Files)

**Location**: `redblackgraph/core/src/redblackgraph/`

4 template files using numpy.distutils templating:

1. **rbg_math.c.src** - AVOS (algebraic operations)
   - Uses `NPY_NO_EXPORT` macro (deprecated)
   - Template variables: @type@, @name@, @utype@
   - Functions: avos_sum, avos_product, MSB

2. **redblack.c.src** - Core red-black graph operations
   - Matrix operations specific to red-black graphs

3. **relational_composition.c.src** - Relational operations
   - Composition operations for relational algebra

4. **warshall.c.src** - Warshall's algorithm
   - Transitive closure computations

**Template System**: numpy.distutils processes @variables@ to generate type-specific code (long, ulong, longlong, etc.)

### Cython Extensions

**Location**: `redblackgraph/sparse/csgraph/`

7 Cython files:

1. **_components.pyx** - Graph connectivity components
2. **_ordering.pyx** - Graph ordering algorithms
3. **_permutation.pyx** - Permutation operations
4. **_rbg_math.pyx** - Red-black graph mathematics (Cython version)
5. **_relational_composition.pyx** - Relational operations (Cython version)
6. **_shortest_path.pyx** - Shortest path algorithms
7. **_tools.pyx** - Utility functions

**Note**: Some functionality exists in both C (.c.src) and Cython (.pyx) forms.

---

## Test Infrastructure

### Test Framework

- **Framework**: pytest
- **Test Command**: `bin/test -u` (runs unit tests)
- **Test Location**: `tests/` directory
- **Test Types**: Primarily unit tests
- **Coverage**: Tracked via codecov

### Test Structure

```
tests/
├── avos/         # AVOS operation tests
├── core/         # Core functionality tests
├── reference/    # Reference implementation tests
├── sparse/       # Sparse matrix tests
└── util/         # Utility tests
```

### Expected Behavior

According to architectural clarification:
- All tests pass on Python 3.6, 3.7, 3.8
- No known test failures
- Test duration: < 5 minutes per Python version
- No flaky tests

**Sprint 1 Status**: Cannot execute tests on new Python versions (3.10+) until Sprint 2 dependency updates complete.

---

## CI/CD Configuration

### Travis CI

**Configuration File**: `.travis.yml`

```yaml
language: python
sudo: true
dist: xenial
python:
  - "3.8"
  - "3.7"
  - "3.6"
install:
  - pip install codecov
  - pip install cython
  - pip install -e ".[test]"
script:
  - bin/test -u
after_success:
  - codecov
```

**Status**: Active on Travis CI
**Platforms**: Linux (xenial dist)
**Python Versions**: 3.8, 3.7, 3.6
**Code Coverage**: Integrated with codecov.io

### Required Updates (Sprint 2)

```yaml
python:
  - "3.12"
  - "3.11"
  - "3.10"
```

---

## Known Issues & Blockers

### Critical Blockers

#### 1. numpy.distutils Deprecation

**Severity**: 🔴 High  
**Impact**: Cannot build on Python 3.10+

**Details**:
- numpy.distutils deprecated since NumPy 1.23.0 (2022)
- Removed in NumPy 2.0
- Python 3.12+ has no distutils in stdlib

**Error**:
```
DeprecationWarning: `numpy.distutils` is deprecated since NumPy 1.23.0
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

**Resolution**: Sprint 2 will update dependency constraints to allow building with compatible NumPy versions. Phase 2 will migrate to Meson.

#### 2. NPY_NO_EXPORT Macro

**Severity**: 🔴 High  
**Impact**: C extension compilation fails

**Details**:
- C code uses `NPY_NO_EXPORT` macro
- Macro may not exist in NumPy 2.0+
- Affects all .c.src template files

**Error**:
```
error: unknown type name 'NPY_NO_EXPORT'; did you mean 'NPY_NO_SMP'?
```

**Resolution**: Sprint 3 will audit and update C API usage.

### Medium Priority Issues

#### 3. dataclasses Backport

**Severity**: 🟡 Medium  
**Impact**: Unnecessary dependency for Python 3.10+

**Details**:
- `dataclasses` backport only needed for Python < 3.7
- Native dataclasses available in Python 3.7+

**Resolution**: Sprint 2 will remove conditional dependency.

#### 4. Loose Version Constraints

**Severity**: 🟡 Medium  
**Impact**: Can install incompatible versions

**Details**:
- No upper bounds on numpy, scipy
- Very old minimum versions (numpy>=0.14.0 from 2015)

**Resolution**: Sprint 2 will add explicit constraints.

---

## Compatibility Matrix

### Current (Master Branch)

| Python | NumPy      | SciPy  | Build | Tests | Notes |
|--------|------------|--------|-------|-------|-------|
| 3.6    | 0.14-1.19  | Any    | ✅    | ✅    | Old NumPy required |
| 3.7    | 0.14-1.19  | Any    | ✅    | ✅    | Old NumPy required |
| 3.8    | 0.14-1.21  | Any    | ✅    | ✅    | Old NumPy required |
| 3.9    | 0.14-1.22  | Any    | ⚠️    | ⚠️    | Not tested in CI |
| 3.10   | Any        | Any    | ❌    | ❌    | numpy.distutils broken |
| 3.11   | Any        | Any    | ❌    | ❌    | numpy.distutils broken |
| 3.12   | Any        | Any    | ❌    | ❌    | numpy.distutils broken |

### Target (After Phase 1)

| Python | NumPy         | SciPy      | Build | Tests | Notes |
|--------|---------------|------------|-------|-------|-------|
| 3.10   | 1.26.0-1.26.x | 1.11.0+    | ✅    | ✅    | Target config |
| 3.11   | 1.26.0-1.26.x | 1.11.0+    | ✅    | ✅    | Target config |
| 3.12   | 1.26.0-1.26.x | 1.11.0+    | ✅    | ✅    | Target config |

---

## Code Patterns Requiring Updates

### 1. Deprecated NumPy Types

**Pattern**: `np.int`, `np.float`, `np.bool`  
**Status**: To be scanned in Task 1.4  
**Resolution**: Sprint 3 - replace with `np.int64`, `np.float64`, `bool`

### 2. NumPy C API

**Pattern**: Various C API calls in .c.src files  
**Status**: To be audited in Task 1.4  
**Resolution**: Sprint 3 - update to NumPy 1.26-compatible API

### 3. Template Processing

**Pattern**: @variable@ templates in .c.src files  
**Status**: Handled by numpy.distutils  
**Resolution**: Phase 2 - migrate to Meson templating

---

## Versioneer Integration

**Status**: Active  
**Version File**: `_version.py` (auto-generated)  
**Configuration**: `versioneer.py` script in root

**Current Behavior**:
- Generates version from git tags
- Version format: `X.Y.Z+hash`

**Phase 1 Impact**:
- Version will bump from 1.x to 2.0.0 (breaking change)
- Versioneer should continue to work unchanged
- Need to verify versioneer compatibility (Task 1.4)

---

## Docker Integration

**Status**: Mentioned in README, not tested in Sprint 1

**Files**:
- `docker/` directory exists
- `.dockerignore` file present

**Phase 1 Requirement**: Docker images should be tested with new Python versions (per architectural clarification).

**Resolution**: Sprint 4 - verify Docker images work with Python 3.10+.

---

## Recommendations for Phase 1

### Immediate Priority (Sprint 2)

1. **Update dependency constraints**:
   ```python
   numpy>=1.26.0,<1.27
   scipy>=1.11.0
   cython>=3.0
   python_requires='>=3.10'
   ```

2. **Remove dataclasses backport**

3. **Use older setuptools** (< 60.0) temporarily to keep numpy.distutils working

4. **Update .travis.yml** to test Python 3.10, 3.11, 3.12

### High Priority (Sprint 3)

1. **Audit C extensions**:
   - Replace `NPY_NO_EXPORT` with appropriate alternative
   - Update any deprecated NumPy C API calls
   - Test compilation with NumPy 1.26

2. **Scan Python code**:
   - Replace `np.int`, `np.float`, `np.bool` with explicit types
   - Check for other NumPy deprecations

3. **Test Cython extensions**:
   - Verify Cython 3.x compatibility
   - Update any Cython-specific deprecations

### Medium Priority (Sprint 4)

1. **Comprehensive testing** on all Python 3.10, 3.11, 3.12
2. **Platform testing** on Linux and macOS
3. **Docker image updates**
4. **Jupyter notebook updates**

---

## Phase 2 Preparation

### Meson Migration Planning

**Required for**: Phase 2 (not Phase 1)

**Template File Processing**:
- Current: numpy.distutils handles @variable@ templates
- Future: Need Meson equivalent or Python preprocessing

**Build Configuration**:
- Current: setup.py with Configuration API
- Future: meson.build files

**C Extension Definition**:
- Current: numpy.distutils.core.Extension
- Future: Meson extension() declarations

---

## Sprint 1 Status

### Completed
- [x] Environment tools verified (uv, gcc)
- [x] Python versions installed (3.10, 3.11, 3.12)
- [x] Virtual environments created
- [x] Build compatibility assessed
- [x] Critical blockers identified

### Blocked
- [ ] Baseline test execution (cannot build on Python 3.10+)
- [ ] Performance baseline (deferred per decision)
- [ ] Dependency version capture (cannot install)

### Findings
- Current codebase cannot build with Python 3.10+
- numpy.distutils deprecation is the main blocker
- C extension compatibility issues identified
- Need Sprint 2 dependency updates before proceeding

---

## Appendices

### A. Build Error Samples

See Sprint 1 implementation work for full error logs showing:
- numpy.distutils deprecation warnings
- distutils.msvccompiler import errors
- NPY_NO_EXPORT compilation errors

### B. Environment Details

**Development Machine**:
- Architecture: ARM64 (aarch64)
- OS: Linux (Ubuntu 24.04)
- Compiler: gcc 13.3.0
- Python Manager: uv 0.9.5

### C. Version Control

- **Git Branch**: master (baseline)
- **Migration Branch**: migration (work branch)
- **Last Commit**: [hash from master]

---

**Report Status**: Complete (with limitations)  
**Limitations**: Could not execute full test suite due to build failures  
**Next Steps**: Proceed with Sprint 2 dependency updates  
**Approval**: Awaiting project owner review

**Last Updated**: 2025-10-21  
**Owner**: QA/Test Lead
