# NumPy 2.x Upgrade - Sprint Plan

**Target Release**: v0.6.0  
**Total Estimated Effort**: 8-10 hours  
**Sprints**: 5  
**Strategy**: Direct removal (no deprecation cycle)

---

## Sprint Overview

| Sprint | Focus Area | Duration | Priority |
|--------|-----------|----------|----------|
| 1 | Remove matrix class | 2 hours | 🔴 Critical |
| 2 | Remove __config__.py | 1 hour | 🔴 Critical |
| 3 | Fix C API deprecations | 2 hours | 🟡 High |
| 4 | Update dependencies & CI | 2 hours | 🟡 High |
| 5 | Testing & validation | 3 hours | 🟢 Required |

---

## Sprint 1: Remove matrix Class

**Duration**: 2 hours  
**Blocker**: Yes - `np.matrix` removed in NumPy 2.0

### Tasks

#### 1.1 Remove matrix class definition
- **File**: `redblackgraph/core/redblack.py`
- **Action**: Delete the entire `matrix` class (lines ~111-115)
  ```python
  class matrix(_Avos, np.matrix):  # DELETE THIS ENTIRE CLASS
      def __new__(cls, data, dtype=None, copy=True):
          return super(matrix, cls).__new__(cls, data, dtype=dtype, copy=copy)
  ```
- **Validation**: Class no longer exists in source

#### 1.2 Update module exports
- **File**: `redblackgraph/core/redblack.py`
- **Action**: Remove `'matrix'` from `__all__` list
- **Validation**: `matrix` not in public API

#### 1.3 Update tests to use array
- **File**: `tests/test_redblack.py`
- **Action**: Replace all `rb.matrix(...)` calls with `rb.array(...)`
- **Count**: 6 occurrences to update
- **Example change**:
  ```python
  # Before
  A = rb.matrix([[-1, 2, 3], [0, -1, 0]])
  
  # After
  A = rb.array([[-1, 2, 3], [0, -1, 0]])
  ```
- **Validation**: All tests pass with `rb.array`

#### 1.4 Check for other usages
- **Action**: Search entire codebase for any remaining `matrix` references
  ```bash
  grep -r "\.matrix\(" --include="*.py"
  grep -r "from.*import.*matrix" --include="*.py"
  ```
- **Update**: Documentation, examples, docstrings

#### 1.5 Update public API documentation
- **Files to check**:
  - `README.md`
  - `docs/` (if exists)
  - Docstrings in `redblackgraph/__init__.py`
- **Action**: Remove any `matrix` references, show only `array` examples

### Exit Criteria
- [ ] `matrix` class deleted from source
- [ ] `'matrix'` removed from `__all__`
- [ ] All tests use `rb.array` instead of `rb.matrix`
- [ ] No `matrix` references in documentation
- [ ] Test suite passes

---

## Sprint 2: Remove __config__.py Generation

**Duration**: 1 hour  
**Blocker**: Yes - `numpy.core.*` imports fail in NumPy 2.0

### Tasks

#### 2.1 Delete generator script
- **File**: `generate_config.py` (root directory)
- **Action**: Delete entire file
- **Rationale**: Not used by package, legacy from numpy.distutils

#### 2.2 Remove from CI workflows
- **Files**:
  - `.github/workflows/ci.yml`
  - `.github/workflows/integration.yml`
  - `.github/workflows/release.yml`
- **Action**: Remove any `python generate_config.py` commands
- **Search for**: `generate_config` in workflow files

#### 2.3 Remove from build configuration
- **File**: `pyproject.toml`
- **Section**: `[tool.cibuildwheel]`
- **Action**: Remove `generate_config.py` from `before-build` steps
- **Example**:
  ```toml
  # Remove this line if present:
  before-build = "python generate_config.py"
  ```

#### 2.4 Verify meson.build
- **File**: `redblackgraph/meson.build`
- **Action**: Confirm `__config__.py` installation is already optional/removed
- **Note**: Analysis shows it's already optional, just verify

#### 2.5 Clean up any generated files
- **Action**: Delete any existing `__config__.py` or `__config__.pyc` files
  ```bash
  find . -name "__config__.py*" -delete
  ```

### Exit Criteria
- [ ] `generate_config.py` deleted
- [ ] No workflow steps call `generate_config.py`
- [ ] No `__config__.py` generation in build process
- [ ] Build succeeds without generating `__config__.py`

---

## Sprint 3: Fix C API Deprecations

**Duration**: 2 hours  
**Blocker**: No (deprecated but still works in NumPy 2.0)

### Tasks

#### 3.1 Replace PyArray_FROM_OF calls
- **File**: `redblackgraph/core/src/redblackgraph/redblackgraphmodule.c`
- **Locations**: Lines 71 and 222
- **Change**:
  ```c
  // Before
  op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
  
  // After
  op[i] = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_ARRAY_ENSUREARRAY, 0);
  ```
- **Note**: `PyArray_FROM_OTF` takes 3 arguments (add `0` for flags parameter)

#### 3.2 Verify no other deprecated C API usage
- **Action**: Search for other deprecated functions:
  ```bash
  grep -r "PyArray_FROM_OF" redblackgraph/core/src/
  grep -r "PyArray_INCREF" redblackgraph/core/src/
  grep -r "PyArray_XDECREF" redblackgraph/core/src/
  ```
- **Fix**: Any other deprecated API calls found

#### 3.3 Check numpy/noprefix.h usage
- **Files**: Multiple C source files
- **Current**: `#include <numpy/noprefix.h>`
- **Decision**: Leave as-is (still supported, not causing issues)
- **Note**: Can be cleaned up in future if it becomes problematic

#### 3.4 Test C extension compilation
- **Action**: Build C extensions with NumPy 2.x
  ```bash
  pip install numpy>=2.0
  python -m pip install --no-build-isolation -e .
  ```
- **Validation**: No compilation warnings or errors

#### 3.5 Test C extension functionality
- **Focus**: Run tests that exercise C extensions
- **Tests**:
  - Warshall operations
  - Relational composition
  - AVOS operations
- **Validation**: All C extension tests pass

### Exit Criteria
- [ ] `PyArray_FROM_OF` replaced with `PyArray_FROM_OTF`
- [ ] No compilation warnings about deprecated API
- [ ] C extensions build successfully with NumPy 2.x
- [ ] All C extension tests pass

---

## Sprint 4: Update Dependencies & CI

**Duration**: 2 hours  
**Blocker**: No (configuration updates)

### Tasks

#### 4.1 Update pyproject.toml dependencies
- **File**: `pyproject.toml`
- **Section**: `[project.dependencies]`
- **Change**:
  ```toml
  dependencies = [
      "numpy>=1.26.0,<3.0",  # Support both NumPy 1.x and 2.x
      "scipy>=1.11.0",       # Already compatible
  ]
  ```

#### 4.2 Update build dependencies
- **File**: `pyproject.toml`
- **Section**: `[build-system.requires]`
- **Change**:
  ```toml
  requires = [
      "meson-python>=0.13.1",
      "Cython>=3.0.0",
      "numpy>=1.26.0,<3.0",  # Match runtime constraint
  ]
  ```

#### 4.3 Update CI test matrix
- **Files**: `.github/workflows/ci.yml`, `.github/workflows/integration.yml`
- **Add NumPy 2.x testing**:
  ```yaml
  strategy:
    matrix:
      python-version: ['3.10', '3.11', '3.12']
      numpy-version: ['1.26', '2.1']
      include:
        - python-version: '3.10'
          numpy-version: '1.26'
        - python-version: '3.10'
          numpy-version: '2.1'
        - python-version: '3.11'
          numpy-version: '1.26'
        - python-version: '3.11'
          numpy-version: '2.1'
        - python-version: '3.12'
          numpy-version: '1.26'
        - python-version: '3.12'
          numpy-version: '2.1'
  ```

#### 4.4 Add NumPy version installation
- **In CI workflows**:
  ```yaml
  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      pip install numpy~=${{ matrix.numpy-version }}
      pip install -e .[test]
  ```

#### 4.5 Update documentation
- **File**: `README.md`
- **Section**: Requirements/Installation
- **Update**: Mention NumPy 2.x support
  ```markdown
  ## Requirements
  - Python >= 3.10
  - NumPy >= 1.26.0 (supports both NumPy 1.x and 2.x)
  - SciPy >= 1.11.0
  ```

#### 4.6 Update CHANGELOG
- **File**: `CHANGELOG.md` or similar
- **Add entry for v0.6.0**:
  ```markdown
  ## [0.6.0] - TBD
  
  ### Added
  - NumPy 2.x support (backwards compatible with NumPy 1.26+)
  
  ### Removed
  - `redblackgraph.matrix` class (deprecated since NumPy 1.19)
    - **Migration**: Use `redblackgraph.array` instead
  - `__config__.py` generation (internal build artifact)
  
  ### Changed
  - Updated C API to use modern NumPy functions
  - Dependency constraint: `numpy>=1.26.0,<3.0`
  ```

### Exit Criteria
- [ ] `pyproject.toml` updated with NumPy <3.0 constraint
- [ ] CI matrix includes NumPy 1.26 and 2.1 testing
- [ ] Documentation reflects NumPy 2.x support
- [ ] CHANGELOG updated with breaking changes

---

## Sprint 5: Testing & Validation

**Duration**: 3 hours  
**Blocker**: No (validation phase)

### Tasks

#### 5.1 Local testing with NumPy 1.26
- **Setup**:
  ```bash
  python -m venv venv-np126
  source venv-np126/bin/activate
  pip install numpy~=1.26.0
  pip install -e .[test]
  ```
- **Tests**:
  ```bash
  pytest -v
  python -m unittest discover tests/
  ```
- **Validation**: All 117 tests pass

#### 5.2 Local testing with NumPy 2.0
- **Setup**:
  ```bash
  python -m venv venv-np20
  source venv-np20/bin/activate
  pip install numpy~=2.0.0
  pip install -e .[test]
  ```
- **Tests**:
  ```bash
  pytest -v
  python -m unittest discover tests/
  ```
- **Validation**: All 117 tests pass

#### 5.3 Local testing with NumPy 2.1 (latest)
- **Setup**:
  ```bash
  python -m venv venv-np21
  source venv-np21/bin/activate
  pip install numpy~=2.1.0
  pip install -e .[test]
  ```
- **Tests**:
  ```bash
  pytest -v
  python -m unittest discover tests/
  ```
- **Validation**: All 117 tests pass

#### 5.4 Test on multiple Python versions
- **Python 3.10**:
  - NumPy 1.26.x ✅
  - NumPy 2.1.x ✅
- **Python 3.11**:
  - NumPy 1.26.x ✅
  - NumPy 2.1.x ✅
- **Python 3.12**:
  - NumPy 1.26.x ✅
  - NumPy 2.1.x ✅

#### 5.5 Platform-specific testing
- **Linux**: Primary development platform
- **Windows**: Test via CI or local VM
- **macOS**: Test via CI or local machine
- **Validation**: C extensions build and tests pass on all platforms

#### 5.6 Integration testing
- **Test scenarios**:
  1. Basic array operations
  2. Warshall transitive closure
  3. Relational composition
  4. AVOS algebra operations
  5. Absorption operations
  6. Indexing and slicing
- **Validation**: Real-world workflows work correctly

#### 5.7 Performance validation
- **Action**: Run basic performance checks
- **Compare**: NumPy 1.26 vs 2.1 performance
- **Note**: NumPy 2.x should be similar or faster
- **Document**: Any significant performance differences

#### 5.8 Documentation testing
- **Test**: All README examples work
- **Test**: Any tutorial/example notebooks
- **Validation**: No broken examples, all use `array` instead of `matrix`

### Exit Criteria
- [ ] All tests pass on NumPy 1.26.x
- [ ] All tests pass on NumPy 2.0.x
- [ ] All tests pass on NumPy 2.1.x
- [ ] All tests pass on Python 3.10, 3.11, 3.12
- [ ] All tests pass on Linux, Windows, macOS
- [ ] Integration tests work correctly
- [ ] Documentation examples are functional

---

## Migration Guide for Users

Create a migration guide document to help users upgrade:

### For Users: Upgrading to v0.6.0

**Breaking Changes:**

1. **`redblackgraph.matrix` removed**
   ```python
   # Before (v0.5.x)
   import redblackgraph as rb
   A = rb.matrix([[1, 2], [3, 4]])
   
   # After (v0.6.0+)
   import redblackgraph as rb
   A = rb.array([[1, 2], [3, 4]])
   ```
   
   **Why?** `np.matrix` was removed from NumPy 2.0. The `array` class provides identical functionality.

2. **NumPy 2.x support**
   - Now supports NumPy 1.26+ and NumPy 2.x
   - No code changes needed if you're already using `rb.array`
   - Performance may improve with NumPy 2.x

**Installation:**
```bash
pip install --upgrade redblackgraph>=0.6.0
pip install --upgrade numpy>=2.0  # Optional but recommended
```

---

## Risk Mitigation

### Risk 1: matrix class users
- **Likelihood**: Low (matrix rarely used)
- **Impact**: Medium (breaking change)
- **Mitigation**: 
  - Clear migration guide
  - Simple one-line fix (matrix → array)
  - Detect usage via grep: `grep -r "rb\.matrix\|redblackgraph\.matrix"`

### Risk 2: C API platform issues
- **Likelihood**: Low (modern API, well-tested)
- **Impact**: High (build failures)
- **Mitigation**: 
  - Test on all platforms before release
  - CI validates all platforms
  - Quick rollback possible

### Risk 3: NumPy 2.x compatibility issues
- **Likelihood**: Very Low (most code already compatible)
- **Impact**: High (broken functionality)
- **Mitigation**: 
  - Extensive testing (Sprint 5)
  - Matrix of Python and NumPy versions in CI
  - Beta testing period

---

## Release Checklist

Before releasing v0.6.0:

- [ ] All sprints completed
- [ ] All exit criteria met
- [ ] CI passing on all platforms/versions
- [ ] CHANGELOG updated
- [ ] Documentation updated
- [ ] Migration guide written
- [ ] Version bumped to 0.6.0
- [ ] Git tag created
- [ ] PyPI release published
- [ ] GitHub release notes posted
- [ ] Announcement sent (if applicable)

---

## Timeline

**Estimated Total**: 2-3 days for development + 1 week for testing/validation

| Phase | Duration | Completion Date |
|-------|----------|-----------------|
| Sprint 1 | 2 hours | Day 1 |
| Sprint 2 | 1 hour | Day 1 |
| Sprint 3 | 2 hours | Day 1 |
| Sprint 4 | 2 hours | Day 2 |
| Sprint 5 | 3 hours | Day 2-3 |
| Beta testing | 1 week | Week 1 |
| Final release | - | Week 2 |

---

## Success Metrics

1. **Compatibility**: Tests pass on NumPy 1.26, 2.0, 2.1
2. **Coverage**: All 117 tests pass (0 failures)
3. **Platforms**: Linux, Windows, macOS all working
4. **Performance**: No regression vs NumPy 1.26
5. **User adoption**: Clean upgrade path, minimal issues

---

## Notes

- **No deprecation cycle**: Going straight to removal as requested
- **Backwards compatibility**: Still supports NumPy 1.26+ for users not ready to upgrade
- **Future-proof**: NumPy <3.0 constraint gives long runway
- **Clean codebase**: Removes legacy `matrix` and `__config__.py` cruft
