# NumPy 2.x Upgrade Analysis

**Date**: 2025-10-22  
**Current NumPy Version**: `>=1.26.0,<2.0`  
**Recommendation**: **Wait for next major release (0.6.0)**

---

## Executive Summary

**TL;DR**: The codebase has **2 blocking issues** and **1 deprecation** that need fixes before NumPy 2.x:

1. 🔴 **BLOCKER**: `np.matrix` inheritance (deprecated, removed in NumPy 2.0)
2. 🔴 **BLOCKER**: `numpy.core.*` imports (private API, removed in NumPy 2.0)  
3. ⚠️ **WARNING**: `PyArray_FROM_OF` C API (deprecated but still works)

**Recommendation**: Fix these issues in a dedicated 0.6.0 release with NumPy 2.x support as the headline feature.

---

## Current State Analysis

### ✅ What's Already Compatible

1. **C Extension Build System**
   - ✅ Uses modern Meson build system
   - ✅ `NPY_NO_DEPRECATED_API NPY_API_VERSION` set correctly
   - ✅ No direct numpy.distutils usage
   - ✅ Cython extensions properly configured

2. **Python Code - Mostly Clean**
   - ✅ Uses `np.ndarray` subclassing (supported)
   - ✅ Standard NumPy API usage (`np.zeros`, `np.asarray`, etc.)
   - ✅ No `numpy.random.RandomState` (legacy random API)
   - ✅ No `numpy.oldnumeric` usage
   - ✅ scipy dependency (>=1.11.0) is NumPy 2.x compatible

3. **C Extensions - Mostly Clean**
   - ✅ Uses modern NumPy C API patterns
   - ✅ Proper `import_array()` initialization
   - ✅ gufunc implementations follow current patterns
   - ✅ No direct pointer manipulation of array internals

---

## 🔴 Blocking Issues

### Issue #1: `np.matrix` Inheritance

**Location**: `redblackgraph/core/redblack.py:111`

```python
class matrix(_Avos, np.matrix):
    def __new__(cls, data, dtype=None, copy=True):
        return super(matrix, cls).__new__(cls, data, dtype=dtype, copy=copy)
```

**Impact**: 
- `np.matrix` was **deprecated** in NumPy 1.19 (2020)
- **REMOVED** in NumPy 2.0
- Used in tests: `tests/test_redblack.py` (6 occurrences)
- Exported in public API: `redblackgraph.core.matrix`

**Usage**:
```python
# tests/test_redblack.py
A = rb.matrix([[-1,  2,  3,  4,  0],
               [ 0, -1,  0,  2,  0],
               ...])
```

**Fix Options**:

1. **Option A: Drop `matrix` class entirely** ⭐ RECOMMENDED
   - Remove `matrix` from `__all__`
   - Add deprecation warning in 0.5.x
   - Remove in 0.6.0
   - Users can use `rb.array` instead (functionally equivalent)

2. **Option B: Reimplement without `np.matrix`**
   - Create custom matrix class inheriting from `np.ndarray`
   - Implement matrix-specific behavior (2D only, `*` operator)
   - More maintenance burden

**Recommendation**: **Option A** - The `matrix` class adds minimal value over `array` class, and removing it simplifies the API. The `array` class already supports all the same operations.

---

### Issue #2: `numpy.core.*` Imports

**Location**: Generated `__config__.py` (line 86)

```python
from numpy.core._multiarray_umath import (
    __cpu_features__, __cpu_baseline__, __cpu_dispatch__
)
```

**Impact**:
- `numpy.core` is a **private module** in NumPy 2.x
- Import will fail in NumPy 2.0+
- This file is **generated** by `generate_config.py`

**Context**:
- `__config__.py` is not actually imported anywhere in the codebase
- It's a legacy file from numpy.distutils days
- We already made it **optional** in `redblackgraph/meson.build`

**Fix**: 
1. **Option A: Remove `__config__.py` generation entirely** ⭐ RECOMMENDED
   - Already optional in meson.build
   - Not used by package code
   - Delete `generate_config.py`
   - Remove from workflow before-build steps

2. **Option B: Update to NumPy 2.x API**
   - Use `numpy.show_config()` instead
   - Update `generate_config.py` to use public API

**Recommendation**: **Option A** - The file serves no purpose and was only kept for compatibility during the numpy.distutils → Meson migration.

---

## ⚠️ Warnings (Non-Blocking)

### Warning #1: `PyArray_FROM_OF` in C Extensions

**Location**: `redblackgraph/core/src/redblackgraph/redblackgraphmodule.c:71, 222`

```c
op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
```

**Impact**:
- Deprecated but **still available** in NumPy 2.0
- Will likely be removed in NumPy 3.0

**Fix**:
- Replace with `PyArray_FROM_OTF()` (modern equivalent)
- Or use `PyArray_CheckExact()` + `PyArray_FROM_OT()`

**Recommendation**: Fix in 0.6.0 while doing other NumPy 2.x updates.

---

### Warning #2: `numpy/noprefix.h` Usage

**Location**: Multiple C files

```c
#include <numpy/noprefix.h>
```

**Impact**:
- Still supported but discouraged
- May be deprecated in future NumPy versions

**Recommendation**: Leave as-is for now, can be cleaned up later. It's not causing issues.

---

## Migration Plan for 0.6.0

### Phase 1: Deprecation (v0.5.2 - Optional)

1. Add deprecation warning to `matrix` class:
   ```python
   import warnings
   class matrix(_Avos, np.matrix):
       def __new__(cls, *args, **kwargs):
           warnings.warn(
               "redblackgraph.matrix is deprecated and will be removed in v0.6.0. "
               "Use redblackgraph.array instead.",
               DeprecationWarning,
               stacklevel=2
           )
           return super().__new__(cls, *args, **kwargs)
   ```

2. Update documentation to recommend `array` over `matrix`

### Phase 2: NumPy 2.x Support (v0.6.0)

**Effort Estimate**: 4-6 hours

#### Step 1: Remove `matrix` class (1 hour)
- [ ] Remove `matrix` class from `redblackgraph/core/redblack.py`
- [ ] Remove `matrix` from `__all__`
- [ ] Update tests to use `array` instead of `matrix`
- [ ] Update any examples/docs

#### Step 2: Remove `__config__.py` generation (30 min)
- [ ] Delete `generate_config.py`
- [ ] Remove `python generate_config.py` from workflows:
  - `.github/workflows/ci.yml`
  - `.github/workflows/integration.yml`  
  - `.github/workflows/release.yml`
  - `pyproject.toml` (cibuildwheel before-build)
- [ ] Update `redblackgraph/meson.build` (already optional, just verify)

#### Step 3: Fix C API deprecations (2 hours)
- [ ] Replace `PyArray_FROM_OF` with `PyArray_FROM_OTF` in:
  - `redblackgraphmodule.c` (2 locations)
- [ ] Test on all platforms

#### Step 4: Update dependencies (30 min)
- [ ] Update `pyproject.toml`:
  ```toml
  dependencies = [
      "numpy>=1.26.0,<3.0",  # Support NumPy 1.x and 2.x
      "scipy>=1.11.0",
  ]
  ```
- [ ] Update build requirements similarly

#### Step 5: Testing (2 hours)
- [ ] Test with NumPy 1.26 (baseline)
- [ ] Test with NumPy 2.0
- [ ] Test with NumPy 2.1 (latest)
- [ ] Run full test suite on all Python versions (3.10, 3.11, 3.12)
- [ ] Test on all platforms (Linux, Windows, macOS)

### Phase 3: CI/CD Updates

Update workflows to test both NumPy 1.x and 2.x:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    numpy-version: ['1.26', '2.1']
```

---

## Dependencies Analysis

### scipy >= 1.11.0
- ✅ **Compatible with NumPy 2.x** (scipy 1.13+ officially supports NumPy 2.0)
- Minimum scipy 1.11.0 works with NumPy 1.x
- Users on NumPy 2.x will automatically get compatible scipy versions

### Cython >= 3.0
- ✅ **Compatible with NumPy 2.x** (Cython 3.0+ supports NumPy 2.0)

---

## Testing Strategy

### Compatibility Testing Matrix

| NumPy Version | Python 3.10 | Python 3.11 | Python 3.12 |
|---------------|-------------|-------------|-------------|
| 1.26.x        | ✅ Current  | ✅ Current  | ✅ Current  |
| 2.0.x         | 🔴 Test     | 🔴 Test     | 🔴 Test     |
| 2.1.x         | 🔴 Test     | 🔴 Test     | 🔴 Test     |

### Test Coverage

1. **Unit Tests**: All 117 tests must pass
2. **C Extension Tests**: Focus on:
   - Warshall operations
   - Relational composition
   - AVOS operations
3. **Array/Matrix Tests**: Verify `array` class works for all use cases
4. **Integration Tests**: End-to-end workflows

---

## Breaking Changes in 0.6.0

### User-Facing Changes

1. **`redblackgraph.matrix` removed**
   - **Migration**: Use `redblackgraph.array` instead
   - **Impact**: LOW (minimal usage, trivial migration)
   - Example:
     ```python
     # Old
     A = rb.matrix([[1, 2], [3, 4]])
     
     # New
     A = rb.array([[1, 2], [3, 4]])
     ```

2. **NumPy 2.x support added**
   - **Impact**: POSITIVE (users can upgrade NumPy)
   - **Compatibility**: Still supports NumPy 1.26+

### Internal Changes

1. `__config__.py` no longer generated
2. C API modernization (`PyArray_FROM_OF` → `PyArray_FROM_OTF`)

---

## Risk Assessment

### Low Risk ✅
- NumPy 1.26+ → 2.x migration is well-documented
- scipy already compatible
- Cython already compatible
- Most code already uses modern APIs

### Medium Risk ⚠️
- `matrix` class removal might affect some users
  - **Mitigation**: Clear migration guide, deprecation warning in 0.5.2
- Platform-specific C API issues
  - **Mitigation**: Test on all platforms before release

### High Risk 🔴
- None identified

---

## Recommendation

### For v0.5.1 (Current Release)
- ✅ **Keep NumPy < 2.0 constraint**
- Continue with current release (already in progress)
- No changes needed

### For v0.5.2 (Optional Deprecation Release)
- Add deprecation warning to `matrix` class
- Update docs to recommend `array` over `matrix`
- Announce NumPy 2.x support coming in 0.6.0

### For v0.6.0 (Major Release - NumPy 2.x)
- Remove `matrix` class
- Remove `__config__.py` generation
- Fix C API deprecations
- Support `numpy>=1.26.0,<3.0`
- **Headline Feature**: "NumPy 2.x Support"
- Release timeline: After v0.5.1 is stable (1-2 weeks)

---

## Resources

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [NumPy 2.0 Release Notes](https://numpy.org/doc/stable/release/2.0.0-notes.html)
- [SciPy NumPy 2.0 Support](https://scipy.github.io/devdocs/dev/core-dev/index.html#numpy-2-0)
- [Cython NumPy Integration](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html)

---

## Conclusion

The migration to NumPy 2.x is **straightforward but requires breaking changes**. The main work is removing the deprecated `matrix` class and `__config__.py` generation. The C API updates are minor.

**Timeline**:
- v0.5.1: Current release (NumPy < 2.0)
- v0.5.2: Optional deprecation warnings (1-2 weeks)
- v0.6.0: NumPy 2.x support (4-6 hours of work, 1-2 weeks after 0.5.1)

This approach ensures:
- No disruption to current users
- Clear migration path
- Thorough testing before NumPy 2.x support
- Professional deprecation cycle
