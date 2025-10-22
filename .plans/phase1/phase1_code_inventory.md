# Phase 1 Code Inventory

**Date**: 2025-10-21  
**Purpose**: Comprehensive inventory of code requiring modification in Phase 1  
**Sprint**: Phase 1, Sprint 1

---

## Executive Summary

This inventory identifies all code, configuration, and build artifacts requiring modification during Phase 1 migration to Python 3.10+ and NumPy 1.26+.

### Key Findings

- **Dataclasses Usage**: 3 files in `types/` module
- **Deprecated NumPy Types**: None found in Python code
- **Setup Files**: 5 active setup.py files (all using numpy.distutils)
- **C Extensions**: 4 template files (.c.src) with NPY_NO_EXPORT usage
- **Cython Extensions**: 7 .pyx files
- **Critical Issues**: Build system incompatibility, C API deprecations

---

## 1. Dataclasses Usage

### Files Using Dataclasses

**Count**: 3 files  
**Location**: `redblackgraph/types/`  
**Complexity**: Low - simple dataclass usage

#### 1.1 `redblackgraph/types/transitive_closure.py`
```python
from dataclasses import dataclass
```
**Usage**: Defines data structures for transitive closure operations  
**Migration**: Remove `from dataclasses import dataclass`, use native Python 3.10+ dataclasses

#### 1.2 `redblackgraph/types/relationship.py`
```python
from dataclasses import dataclass
```
**Usage**: Defines relationship data structures  
**Migration**: Already uses standard library, no change needed (just verify import)

#### 1.3 `redblackgraph/types/ordering.py`
```python
from dataclasses import dataclass
```
**Usage**: Defines ordering data structures  
**Migration**: Already uses standard library, no change needed (just verify import)

### Assessment

**Complexity**: ⭐ Low  
**Risk**: 🟢 Low  
**Effort**: < 1 hour

**Action Items**:
- Verify all three files import from standard library `dataclasses` (not backport)
- Remove conditional dependency from setup.py
- Test that dataclasses behave identically on Python 3.10+

---

## 2. NumPy Deprecated Patterns

### Python Code Scan Results

**Deprecated np.int, np.float, np.bool**: ✅ None found

**Scan Command**:
```bash
grep -r "np\.int[^a-z_0-9]\|np\.float[^a-z_0-9]\|np\.bool[^a-z_0-9]" redblackgraph/ --include="*.py"
```

**Result**: No matches

**Assessment**: Python code is already using explicit NumPy dtypes or Python builtins.

### NumPy Random API

**Status**: Not scanned yet  
**Action**: Sprint 3 - check for `numpy.random.RandomState` vs new Generator API

---

## 3. Setup.py Files

### All Setup Files (5 total)

All files use **numpy.distutils** Configuration API and must be updated or replaced.

#### 3.1 Root: `./setup.py`

**Lines**: 179  
**Purpose**: Main package configuration and build orchestration

**Key Sections**:
```python
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from numpy.distutils.command.build_ext import build_ext
```

**Features**:
- Versioneer integration for version management
- Custom build_ext with GNU linker version scripts
- Calls `tools/cythonize.py` to generate Cython code
- Configures all subpackages

**Dependencies**:
```python
install_requires=[
    'dataclasses;python_version<"3.7"',  # ← Remove
    'numpy>=0.14.0',                      # ← Update to >=1.26.0,<1.27
    'scipy',                              # ← Update to >=1.11.0
    'XlsxWriter',                         # ← Keep
    'fs-crawler>=0.3.2'                   # ← Keep
]

setup_requires=[
    'numpy>=0.18.1',                      # ← Update to >=1.26.0,<1.27
    'cython'                              # ← Update to >=3.0
]

python_requires='>=3.6'                   # ← Update to >=3.10
```

**Classifiers to Update**:
```python
classifiers=[
    # Remove these:
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    
    # Keep/Update:
    'Programming Language :: Python :: 3.10',
    
    # Add these:
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
]
```

**Sprint 2 Updates**:
- Update all version constraints
- Remove dataclasses conditional
- Update Python classifiers
- Add setuptools<60.0 temporarily (for numpy.distutils compatibility)

#### 3.2 Package: `./redblackgraph/setup.py`

**Lines**: 19  
**Purpose**: Package hierarchy configuration

```python
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None):
    config = Configuration('redblackgraph',parent_package,top_path)
    config.add_subpackage('core')
    config.add_subpackage('sparse')
    config.add_subpackage('reference')
    config.add_subpackage('types')
    config.add_subpackage('util')
    return config
```

**Sprint 2 Updates**: None (works with Phase 1 numpy.distutils)  
**Phase 2**: Replace with Meson configuration

#### 3.3 Core: `./redblackgraph/core/setup.py`

**Purpose**: C extension module configuration for .c.src template files

**Key Features**:
- Processes 4 .c.src template files
- Generates type-specific C code
- Links against NumPy C API

**Sprint 2 Updates**: None  
**Sprint 3**: May need C code updates for NumPy 1.26 API  
**Phase 2**: Replace with Meson build

#### 3.4 Sparse: `./redblackgraph/sparse/setup.py`

**Purpose**: Sparse matrix extension configuration

**Sprint 2 Updates**: None  
**Phase 2**: Replace with Meson build

#### 3.5 CSGraph: `./redblackgraph/sparse/csgraph/setup.py`

**Purpose**: Cython extension module configuration for 7 .pyx files

**Sprint 2 Updates**: None  
**Sprint 3**: Verify Cython 3.x compatibility  
**Phase 2**: Replace with Meson build

---

## 4. C Extensions (.c.src Template Files)

### Overview

**Location**: `redblackgraph/core/src/redblackgraph/`  
**Count**: 4 files  
**Technology**: numpy.distutils template system  
**Issue**: Use deprecated `NPY_NO_EXPORT` macro

### 4.1 `rbg_math.c.src`

**Purpose**: AVOS (Algebraic Value Operations System) mathematics

**Template Variables**:
- `@type@`: npy_byte, npy_short, npy_int, npy_long, npy_longlong, etc.
- `@name@`: byte, short, int, long, longlong, etc.
- `@utype@`: unsigned equivalents

**Functions Generated** (per type):
```c
NPY_NO_EXPORT @type@ @name@_avos_sum(@type@ a, @type@ b)
NPY_NO_EXPORT short @name@_MSB(@type@ x)
NPY_NO_EXPORT @utype@ @name@_avos_product(@type@ lhs, @type@ rhs)
```

**NumPy C API Usage**:
- `NPY_NO_EXPORT` - Deprecated/removed in modern NumPy
- `NPY_NO_DEPRECATED_API NPY_API_VERSION` - Good practice
- `npy_*` types - Standard NumPy C types
- `npy_3kcompat.h` - Python 2/3 compatibility (can remove)

**Issues**:
1. ❌ `NPY_NO_EXPORT` not defined in modern NumPy
2. ⚠️ `npy_3kcompat.h` is for Python 2 compat (unnecessary in Python 3.10+)

**Sprint 3 Actions**:
- Replace `NPY_NO_EXPORT` with `static` or remove
- Remove `npy_3kcompat.h` include
- Test compilation with NumPy 1.26

### 4.2 `redblack.c.src`

**Purpose**: Core red-black graph operations

**NumPy C API Usage**:
- Similar patterns to rbg_math.c.src
- Matrix manipulation functions
- Uses npy_ types

**Sprint 3**: Audit and update similar to rbg_math.c.src

### 4.3 `relational_composition.c.src`

**Purpose**: Relational algebra operations

**Special Features**:
- Uses einsum-related NumPy internals
- `NPY_EINSUM_DBG_*` macros for debugging
- `NPY_BEGIN_THREADS_NDITER` for threading

**NumPy C API Usage**:
```c
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#ifdef NPY_ALLOW_THREADS
#define NPY_BEGIN_THREADS_NDITER(iter)
```

**Types Used**:
- `npy_byte`, `npy_short`, `npy_int`, `npy_long`, `npy_longlong`
- `npy_ubyte`, `npy_ushort`, `npy_uint`, `npy_ulong`, `npy_ulonglong`
- `npy_intp` (index type)

**Sprint 3**: Verify einsum internal API stability with NumPy 1.26

### 4.4 `warshall.c.src`

**Purpose**: Warshall's algorithm (transitive closure)

**NumPy C API Usage**:
- Similar template patterns
- Graph algorithm implementations

**Sprint 3**: Audit and update

---

## 5. Cython Extensions (.pyx files)

### Overview

**Location**: `redblackgraph/sparse/csgraph/`  
**Count**: 7 files  
**Technology**: Cython (need to verify version compatibility)

### Files

1. **`_components.pyx`** - Graph connectivity components
2. **`_ordering.pyx`** - Graph ordering algorithms
3. **`_permutation.pyx`** - Permutation operations
4. **`_rbg_math.pyx`** - AVOS math (Cython implementation)
5. **`_relational_composition.pyx`** - Relational ops (Cython implementation)
6. **`_shortest_path.pyx`** - Shortest path algorithms
7. **`_tools.pyx`** - Utility functions

### Note: Dual Implementations

Some functionality exists in both C (.c.src) and Cython (.pyx):
- **rbg_math**: Both `rbg_math.c.src` and `_rbg_math.pyx`
- **relational_composition**: Both `.c.src` and `.pyx` versions

**Question**: Are both used, or is one preferred? Need to investigate module imports.

### Cython Compatibility

**Current**: No explicit Cython version constraint  
**Target**: Cython >= 3.0  
**Available**: Cython 3.1.5

**Sprint 3 Actions**:
- Test compilation with Cython 3.1.5
- Check for Cython 3.x breaking changes
- Verify NumPy integration works

---

## 6. Python Version-Specific Code

### Conditionals

**Scan Result**: No Python version conditionals found in main code

**setup.py**:
```python
'dataclasses;python_version<"3.7"'  # ← Only instance
```

**Assessment**: Minimal version-specific code, clean migration path

---

## 7. NumPy C API Version

### Current API Usage

**Defines**:
```c
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
```

**Interpretation**: Code requests to use non-deprecated NumPy API at compile time. This is good practice.

**Issue**: The specific NumPy API version is determined by installed NumPy version, not specified in code.

### NumPy 1.26 C API Changes

**Sprint 3 Actions**:
1. Review NumPy 1.26 C API changelog
2. Identify breaking changes affecting redblackgraph
3. Test compilation against NumPy 1.26
4. Update code as needed

**Known Issues**:
- `NPY_NO_EXPORT` macro removed/unavailable
- Potential changes to array iteration API
- Potential changes to einsum internals

---

## 8. einsum Usage

### Risk Assessment

**Location**: `redblackgraph/core/src/redblackgraph/relational_composition.c.src`

**Usage**: Code uses internal NumPy einsum macros:
```c
#define NPY_EINSUM_DBG_TRACING 0
#define NPY_EINSUM_DBG_PRINT(s)
```

**Risk**: 🟡 Medium

**Concern** (from architectural clarification):
> "Yes, we use einsum over internal numpy matrix representation"

**Implication**: Code may depend on NumPy internal structures that could change.

**Sprint 3 Actions**:
1. Identify exactly how einsum internals are used
2. Check NumPy 1.26 einsum implementation for changes
3. Test functionality extensively
4. Consider using public NumPy API if internals have changed

---

## 9. Versioneer

### Current Status

**Files**:
- `versioneer.py` (68,611 bytes) - Large standalone script
- `_version.py` (auto-generated)

**Integration**: setup.py imports and uses versioneer

**Version**: Unknown (could not easily determine)

**Sprint 1 Action**: Check for versioneer updates

### Checking Versioneer Version

```bash
# From versioneer.py file
head -50 versioneer.py | grep -i version
```

**Latest Versioneer**: 0.29 (as of 2024)

**Sprint 2 Actions**:
1. Identify current versioneer version
2. Check if update available
3. If update available and safe, update versioneer
4. Test version generation works correctly

**Risk**: 🟢 Low - Versioneer is stable and rarely breaks

---

## 10. CI/CD Configuration

### Travis CI: `.travis.yml`

**Current Configuration**:
```yaml
python:
  - "3.8"
  - "3.7"
  - "3.6"
```

**Sprint 2 Update**:
```yaml
python:
  - "3.12"
  - "3.11"
  - "3.10"
```

**Additional Changes**:
- May need to update `dist:` (currently xenial)
- Ensure Cython installed before build
- Verify codecov integration still works

---

## 11. Documentation Files

### README.md

**Current Python Badges**:
```markdown
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)]
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)]
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)]
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)]
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)]
```

**Sprint 2 Update**:
```markdown
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)]
```

### Jupyter Notebooks

**Location**: `notebooks/` directory

**Sprint 4 Action**: Update notebooks to reflect Python 3.10+ requirements

---

## 12. Docker Configuration

### Files

- `docker/` directory (contents not detailed in Sprint 1)
- `.dockerignore`

**Sprint 4 Action**: Update Docker images to use Python 3.10+

---

## Summary Statistics

| Category | Count | Status |
|----------|-------|--------|
| Dataclasses imports | 3 | ✅ Easy to migrate |
| Deprecated np types | 0 | ✅ Already clean |
| Setup.py files | 5 | ⚠️ All need review |
| C template files | 4 | ❌ Need C API updates |
| Cython files | 7 | ⚠️ Need Cython 3.x test |
| Version conditionals | 1 | ✅ Easy to remove |
| CI config files | 1 | ⚠️ Needs update |
| Documentation files | Multiple | ⚠️ Needs update |

---

## Modification Complexity Assessment

### Low Complexity (< 1 hour each)
- Remove dataclasses conditional from setup.py
- Update Python version classifiers
- Update README.md badges
- Update .travis.yml

### Medium Complexity (2-4 hours each)
- Update dependency version constraints in setup.py
- Test Cython 3.x compatibility
- Update versioneer (if needed)

### High Complexity (4-8 hours each)
- Fix NPY_NO_EXPORT in C extensions
- Audit and update NumPy C API usage
- Test einsum internal API changes
- Full integration testing

---

**Document Status**: Complete  
**Next Deliverable**: Modification Checklist  
**Last Updated**: 2025-10-21  
**Owner**: Tech Lead
