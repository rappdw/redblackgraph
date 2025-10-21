# RedBlackGraph Library Modernization: Migration Specification

**Document Version**: 1.0  
**Last Updated**: October 21, 2025  
**Status**: Planning

---

## Executive Summary

This migration specification outlines the comprehensive plan to modernize the RedBlackGraph library to align with current Python ecosystem standards. The migration addresses critical deprecations in NumPy's build infrastructure while upgrading to modern Python versions and dependencies.

### Critical Driver
The **removal of `numpy.distutils` in NumPy 2.0** necessitates a complete build system migration. NumPy 1.26.x is the last version supporting legacy build tools and serves as our transition point.

### Primary Objectives
1. Migrate from deprecated `numpy.distutils` to modern Meson build system
2. Upgrade Python support from 3.6+ to 3.10+
3. Upgrade NumPy from >=0.14.0 to 2.0+
4. Upgrade SciPy to 1.11+ with explicit version constraints
5. Modernize CI/CD infrastructure from Travis CI to GitHub Actions

### Timeline
**Conservative Estimate**: 4-5 weeks of focused development  
**Aggressive Estimate**: 2-3 weeks (higher risk)

---

## Current State Analysis

### Version Dependencies
| Component | Current | Issues |
|-----------|---------|--------|
| Python | >=3.6 | Python 3.6-3.9 are EOL or near EOL |
| NumPy | >=0.14.0 | Extremely outdated (2017), no upper bound |
| SciPy | None | No version constraint at all |
| Cython | Unspecified | Need Cython 3.0+ for NumPy 2.0 |

### Build System
- **Technology**: numpy.distutils (DEPRECATED in NumPy 1.26, REMOVED in NumPy 2.0)
- **C Extensions**: Multiple with `.c.src` templating (redblackgraphmodule.c)
- **Cython Extensions**: Multiple `.pyx` files in `sparse/csgraph/`
- **Generated Code**: Sparsetools generation via `generate_sparsetools.py`
- **Build Files**: Nested `setup.py` files in multiple packages

### Critical Dependencies
- `numpy.distutils.misc_util.Configuration`
- `numpy.distutils.command.build_ext`
- `numpy.distutils.core.setup`
- NumPy C API for C extensions
- NumPy `.src` templating system for code generation
- SciPy sparse matrix APIs (csr_matrix, sputils)

### Testing Infrastructure
- **CI Platform**: Travis CI on Ubuntu Xenial
- **Python Versions**: 3.6, 3.7, 3.8
- **Coverage**: codecov integration

---

## Migration Phases

### Phase 1: Intermediate Upgrade (Python 3.10, NumPy 1.26, SciPy 1.11)

**Goal**: Establish a stable baseline with modern dependencies while maintaining numpy.distutils compatibility.

**Duration**: 1 week (Conservative) / 3 days (Aggressive)

#### Technical Requirements

**1.1 Dependency Updates**
- Python: `>=3.10` (drop 3.6-3.9)
- NumPy: `>=1.26.0,<2.0.0` (last version with numpy.distutils)
- SciPy: `>=1.11.0` (compatible with NumPy 1.26+)
- Cython: `>=0.29.21` (preparation for Cython 3.0)

**1.2 Files to Modify**

`setup.py`:
```python
python_requires='>=3.10'
install_requires=[
    'numpy>=1.26.0,<2.0.0',
    'scipy>=1.11.0',
    'XlsxWriter',
    'fs-crawler>=0.3.2',
]
setup_requires=['numpy>=1.26.0,<2.0.0']
# Remove: dataclasses;python_version<"3.7"
classifiers=[
    # Remove: Python 3.5-3.9
    # Add: Python 3.10, 3.11, 3.12
]
```

`requirements.txt`:
```
numpy>=1.26.0,<2.0.0
scipy>=1.11.0
XlsxWriter
fs-crawler>=0.3.2
# Remove: dataclasses; python_version < '3.7'
```

`.travis.yml`:
```yaml
dist: jammy  # Ubuntu 22.04
python:
  - "3.10"
  - "3.11"
  - "3.12"
```

**1.3 Code Changes**
- Remove `dataclasses` backport imports (built-in since Python 3.7)
- Review and address deprecation warnings from NumPy 1.26
- Update README.md badges and installation instructions

#### Acceptance Criteria
- ✅ All tests pass on Python 3.10, 3.11, 3.12
- ✅ All tests pass with NumPy 1.26.x and SciPy 1.11+
- ✅ No critical deprecation warnings
- ✅ Wheels build successfully with current build system
- ✅ Package installable via `pip install -e .`
- ✅ Command-line tool `rbg` functions correctly

#### Testing Checklist
- [ ] Install in clean virtual environment
- [ ] Build C extensions: `python setup.py build_ext --inplace`
- [ ] Run full test suite: `pytest tests/ -v --cov=redblackgraph`
- [ ] Test matrix operations
- [ ] Test sparse matrix operations
- [ ] Test csgraph operations
- [ ] Test CLI tool: `rbg --help`
- [ ] Build and test wheel: `python setup.py bdist_wheel`
- [ ] Test on Python 3.10, 3.11, 3.12

#### Rollback Strategy
If Phase 1 fails, revert to current state maintaining Python 3.6+ support and NumPy <1.26.

---

### Phase 2: Build System Migration (Meson + meson-python)

**Goal**: Replace numpy.distutils with Meson build system, enabling future NumPy 2.0 upgrade.

**Duration**: 2-3 weeks (Conservative) / 1-2 weeks (Aggressive)

#### Technical Requirements

**2.1 New Build Infrastructure**

Create `pyproject.toml`:
```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "Cython>=3.0.0",
    "numpy>=1.26.0",
]
build-backend = "mesonpy"

[project]
name = "RedBlackGraph"
dynamic = ["version"]
description = "Red Black Graph - A DAG of Multiple, Interleaved Binary Trees"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "AGPLv3+"}
authors = [
    {name = "Daniel Rapp", email = "rappdw@gmail.com"}
]
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pylint>=2.15",
]

[project.scripts]
rbg = "redblackgraph.__main__:main"
```

**2.2 Meson Build Files**

Create `meson.build` (root):
```meson
project(
  'redblackgraph',
  'c', 'cython',
  version: '0.5.0',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
cc = meson.get_compiler('c')

subdir('redblackgraph')
```

Create `redblackgraph/core/meson.build`:
```meson
# Preprocess .c.src template files
conv_template = find_program('../../tools/conv_template.py')

generated_sources = [
  custom_target('rbg_math_c',
    input: 'src/redblackgraph/rbg_math.c.src',
    output: 'rbg_math.c',
    command: [conv_template, '@INPUT@', '@OUTPUT@']),
  custom_target('redblack_c',
    input: 'src/redblackgraph/redblack.c.src',
    output: 'redblack.c',
    command: [conv_template, '@INPUT@', '@OUTPUT@']),
  custom_target('relational_composition_c',
    input: 'src/redblackgraph/relational_composition.c.src',
    output: 'relational_composition.c',
    command: [conv_template, '@INPUT@', '@OUTPUT@']),
  custom_target('warshall_c',
    input: 'src/redblackgraph/warshall.c.src',
    output: 'warshall.c',
    command: [conv_template, '@INPUT@', '@OUTPUT@']),
]

# C extension module
py.extension_module(
  '_redblackgraph',
  ['src/redblackgraph/redblackgraphmodule.c'] + generated_sources,
  include_directories: [inc_np, include_directories('src/redblackgraph')],
  dependencies: [py_dep],
  install: true,
  subdir: 'redblackgraph/core'
)

# Install Python files
py.install_sources(
  ['__init__.py', 'avos.py', 'redblack.py'],
  subdir: 'redblackgraph/core'
)
```

Create `redblackgraph/sparse/csgraph/meson.build`:
```meson
# Cython extensions
cython_extensions = [
  '_shortest_path',
  '_rbg_math',
  '_components',
  '_permutation',
  '_ordering',
  '_relational_composition',
  '_tools',
]

foreach ext : cython_extensions
  py.extension_module(
    ext,
    ext + '.pyx',
    include_directories: [inc_np],
    dependencies: [py_dep],
    install: true,
    subdir: 'redblackgraph/sparse/csgraph',
    override_options: ['cython_language=c'],
  )
endforeach

py.install_sources(
  ['__init__.py', '_validation.py'],
  subdir: 'redblackgraph/sparse/csgraph'
)
```

**2.3 Template Processing Strategy**

Since Meson doesn't support NumPy's `.c.src` templating, select one approach:

**Option A: Create conv_template.py** (Recommended for Phase 2)
- Implement NumPy's template syntax parser
- Generate `.c` files during build
- Minimal code changes required

**Option B: Convert to Cython**
- Rewrite templated C as Cython
- Better long-term maintainability
- Requires more code changes

**Option C: Expand to plain C**
- Generate all type variants manually
- No build-time dependencies
- Increased code size

**Recommendation**: Start with Option A, evaluate Option B for future work.

**2.4 Files to Remove** (after successful migration)
- `setup.py` (root)
- `redblackgraph/setup.py`
- `redblackgraph/core/setup.py`
- `redblackgraph/sparse/setup.py`
- `redblackgraph/sparse/csgraph/setup.py`
- Potentially `setup.cfg`

**2.5 Update Supporting Infrastructure**
- Update `bin/build-linux-wheels.sh` for Meson
- Update `bin/build-osx-wheels.sh` for Meson
- Update Docker files in `docker/`
- Update `.gitignore` for Meson artifacts

#### Acceptance Criteria
- ✅ Meson build completes successfully
- ✅ All C extensions compile
- ✅ All Cython extensions compile
- ✅ All tests pass with Meson-built package
- ✅ Wheels build: `python -m build`
- ✅ Editable install works: `pip install -e .`
- ✅ Build succeeds on Linux and macOS
- ✅ Zero numpy.distutils dependencies

#### Testing Checklist
- [ ] Clean build: `rm -rf build/ builddir/`
- [ ] Build package: `python -m build`
- [ ] Install editable: `pip install -e . --no-build-isolation`
- [ ] Run full test suite
- [ ] Test wheel installation in clean environment
- [ ] Verify all imports work
- [ ] Test on Python 3.10, 3.11, 3.12
- [ ] Test on Linux
- [ ] Test on macOS

#### Rollback Strategy
If Phase 2 fails, rollback to Phase 1. Can maintain NumPy 1.26 with numpy.distutils for security updates.

---

### Phase 3: NumPy 2.0 Upgrade

**Goal**: Upgrade to NumPy 2.0+ after successful Meson migration.

**Duration**: 1 week (Conservative) / 3 days (Aggressive)

#### Technical Requirements

**3.1 Dependency Updates**

Update `pyproject.toml`:
```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "Cython>=3.0.0",
    "numpy>=2.0.0",
]

[project]
dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.13.0",  # Ensure NumPy 2.0 compatibility
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]
```

**3.2 C Extension Updates**

Update C files to use NumPy 2.0 C API:
```c
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

// Review all NumPy C API calls
// Update array creation/access patterns
// Check dtype handling
```

Key changes to review:
- Array creation functions
- Data type handling
- Memory management
- API compatibility

**3.3 Cython Extension Updates**

Update `.pyx` files for NumPy 2.0:
```cython
# cython: language_level=3

import numpy as np
cimport numpy as cnp

# Ensure numpy is initialized
cnp.import_array()

# Review buffer syntax and typed memory views
def my_function(cnp.float64_t[:, ::1] arr):
    # Prefer buffer syntax for NumPy 2.0
    pass
```

**3.4 Python Code Updates**

- Check for deprecated NumPy functions
- Update dtype specifications to be explicit
- Review array creation patterns
- Check NumPy scalar type changes

**3.5 Known NumPy 2.0 Changes**
- Improved ABI stability
- Some functions removed/renamed
- Stricter dtype handling
- Changed default behaviors
- Performance improvements

#### Acceptance Criteria
- ✅ All tests pass with NumPy 2.0+
- ✅ No runtime deprecation warnings
- ✅ Performance comparable or better
- ✅ Memory usage acceptable
- ✅ Reference counting correct
- ✅ Compatible with latest SciPy

#### Testing Checklist
- [ ] Clean rebuild: `pip install "numpy>=2.0.0"`
- [ ] Run full test suite with verbose output
- [ ] Check for warnings
- [ ] Performance benchmarking
- [ ] Memory profiling
- [ ] Test with SciPy 1.13.0+
- [ ] Validate on Python 3.10, 3.11, 3.12

#### Rollback Strategy
If Phase 3 fails, rollback to Phase 2. Meson build with NumPy 1.26 is stable long-term.

---

### Phase 4: CI/CD Modernization (GitHub Actions)

**Goal**: Migrate from Travis CI to GitHub Actions.

**Duration**: 3-5 days

#### Technical Requirements

**4.1 GitHub Actions Workflows**

Create `.github/workflows/test.yml`:
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=redblackgraph --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

Create `.github/workflows/build.yml`:
```yaml
name: Build Wheels

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Build wheel
      run: |
        pip install build
        python -m build
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
        path: dist/*
```

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

**4.2 Additional Improvements**
- Add Dependabot for dependency updates
- Consider pre-commit hooks (black, ruff, mypy)
- Update codecov integration

**4.3 Cleanup**
- Remove `.travis.yml` after verification
- Update README badges
- Update contributing docs

#### Acceptance Criteria
- ✅ GitHub Actions run successfully
- ✅ Tests pass on all configurations
- ✅ Coverage uploads to codecov
- ✅ Wheel building works
- ✅ Publishing workflow tested

#### Testing Checklist
- [ ] Verify workflows trigger correctly
- [ ] Check test matrix coverage
- [ ] Validate artifact uploads
- [ ] Test badge updates
- [ ] Dry-run PyPI publishing

---

## Risk Assessment & Mitigation

### High Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Build system migration breaks compilation | Critical | Medium | Phased approach, extensive testing, maintain Phase 1 fallback |
| NumPy 2.0 C API incompatibilities | High | Medium | Thorough review of migration guide, test on dev branch |
| Template preprocessing fails | High | Medium | Create robust conv_template.py, consider Cython conversion |
| Platform-specific build issues | Medium | High | Test on multiple platforms early, use CI matrix |

### Medium Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| SciPy API changes | Medium | Low | SciPy sparse API is stable, verify during testing |
| Performance regressions | Medium | Low | Benchmark before/after, profile critical paths |
| CI/CD migration issues | Low | Medium | Run parallel with Travis initially |

### Low Risk Items

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Python 3.10+ compatibility | Low | Low | Python maintains good backward compatibility |
| Documentation gaps | Low | High | Incremental documentation updates |
| Dependency conflicts | Low | Low | Use virtual environments, specify version ranges |

---

## Implementation Strategy

### Branching Strategy
```
main (stable production)
├── phase1-upgrade (Python 3.10, NumPy 1.26, SciPy 1.11)
│   └── phase2-meson (Meson build system)
│       └── phase3-numpy2 (NumPy 2.0)
└── phase4-ci (GitHub Actions - can branch from any phase)
```

### Testing Gates

Each phase must pass these gates before proceeding:

1. **All existing tests pass**
2. **No critical warnings**
3. **Successful wheel build**
4. **Clean environment installation test**
5. **Manual smoke tests of key features**
6. **Performance within 10% of baseline**

### Validation Process

**Import Test**:
```python
import redblackgraph
from redblackgraph.core import redblack
from redblackgraph.sparse import rb_matrix
from redblackgraph.sparse.csgraph import (
    transitive_closure,
    all_pairs_avos_path,
    components,
)
print("✓ All imports successful")
```

**Build Verification** (each phase):
```bash
# Clean environment
python -m venv test_env
source test_env/bin/activate

# Install and test
pip install -e ".[test]"
pytest tests/ -v --cov=redblackgraph

# Test wheel
pip install build
python -m build
pip install dist/RedBlackGraph-*.whl
python -c "import redblackgraph; print(redblackgraph.__version__)"
```

---

## Success Metrics

### Phase 1 Success Criteria
- All tests pass on Python 3.10, 3.11, 3.12
- All tests pass with NumPy 1.26.x and SciPy 1.11+
- Wheels build successfully
- Zero critical deprecation warnings
- Package installs cleanly via pip

### Phase 2 Success Criteria
- Meson build completes without errors
- All extensions compile successfully
- All tests pass with Meson-built package
- Wheels build with Meson
- Zero numpy.distutils dependencies
- Successful builds on Linux and macOS

### Phase 3 Success Criteria
- All tests pass with NumPy 2.0+
- No runtime warnings or errors
- Performance within 5% of NumPy 1.26 baseline
- Memory usage acceptable
- Documentation fully updated

### Phase 4 Success Criteria
- GitHub Actions workflows operational
- Tests pass in CI on all configurations
- Coverage reporting functional
- Wheel building automated

---

## Timeline & Resource Allocation

### Conservative Timeline (Recommended)

| Phase | Duration | Calendar Time | Effort |
|-------|----------|---------------|--------|
| Phase 1: Initial Upgrade | 5-7 days | Week 1 | Full-time |
| Phase 2: Meson Migration | 15-21 days | Weeks 2-4 | Full-time |
| Phase 3: NumPy 2.0 | 5-7 days | Week 5 | Full-time |
| Phase 4: CI/CD | 3-5 days | Week 5-6 | Part-time |
| **Total** | **28-40 days** | **5-6 weeks** | |

### Aggressive Timeline

| Phase | Duration | Calendar Time | Effort |
|-------|----------|---------------|--------|
| Phase 1: Initial Upgrade | 3 days | Days 1-3 | Full-time |
| Phase 2: Meson Migration | 7-10 days | Days 4-13 | Full-time |
| Phase 3: NumPy 2.0 | 3 days | Days 14-16 | Full-time |
| Phase 4: CI/CD | 2-3 days | Days 17-19 | Part-time |
| **Total** | **15-19 days** | **2.5-3 weeks** | |

### Milestone Deliverables

**Milestone 1** (End of Phase 1):
- Release v0.5.1 with Python 3.10+, NumPy 1.26, SciPy 1.11
- Tagged commit: `v0.5.1-numpy126`
- Updated documentation

**Milestone 2** (End of Phase 2):
- Release v0.6.0 with Meson build system
- Tagged commit: `v0.6.0-meson`
- Migration guide for developers

**Milestone 3** (End of Phase 3):
- Release v1.0.0 with NumPy 2.0 support
- Tagged commit: `v1.0.0-numpy2`
- Performance benchmarks published

**Milestone 4** (End of Phase 4):
- Release v1.0.1 with GitHub Actions
- Tagged commit: `v1.0.1-gha`
- CI/CD documentation

---

## Open Questions & Decisions

### Questions Requiring Resolution

1. **Python 3.9 Support**: Should we support Python 3.9 (EOL October 2025)?
   - **Recommendation**: Drop for simplicity, target Python 3.10+ only

2. **NumPy 2.0 Timing**: Target immediately after Meson migration or wait for ecosystem?
   - **Recommendation**: Proceed after Meson stable, but maintain NumPy 1.26 branch

3. **Numba Support**: Keep `requirements-numba.txt`?
   - **Recommendation**: Evaluate usage statistics, likely deprecate if unused

4. **Template Strategy**: Which approach for `.c.src` files?
   - **Recommendation**: Start with conv_template.py, evaluate Cython conversion in Phase 3

5. **Version Numbering**: Should NumPy 2.0 support be v1.0.0 or continue from current?
   - **Recommendation**: v1.0.0 to signal major milestone and breaking changes

6. **Windows Support**: Should we test and support Windows builds?
   - **Recommendation**: Optional, add if community requests, focus on Linux/macOS first

### Key Decisions to Document

- [ ] Final Python version support policy (3.10+ confirmed)
- [ ] NumPy/SciPy version support matrix
- [ ] Template processing approach selection
- [ ] Release cadence post-migration
- [ ] Long-term support commitments for v1.0

---

## References & Resources

### Official Documentation
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Meson Build System](https://mesonbuild.com/)
- [Meson Python Documentation](https://meson-python.readthedocs.io/)
- [SciPy Build Documentation](https://docs.scipy.org/doc/scipy/building/)
- [Cython 3.0 Documentation](https://cython.readthedocs.io/)
- [Python Packaging User Guide](https://packaging.python.org/)

### Example Projects
- [NumPy's meson.build files](https://github.com/numpy/numpy/tree/main)
- [SciPy's Meson migration](https://github.com/scipy/scipy) (reference implementation)
- [scikit-learn build system](https://github.com/scikit-learn/scikit-learn)

### Tools & Utilities
- [cibuildwheel](https://cibuildwheel.readthedocs.io/) - Cross-platform wheel building
- [pyproject.toml validator](https://validate-pyproject.readthedocs.io/)
- [Meson migration examples](https://github.com/mesonbuild/meson-python/tree/main/docs/how-to-guides)

### Related Reading
- [From numpy.distutils to Meson](https://labs.quansight.org/blog/2021/07/moving-scipy-to-meson)
- [NumPy 2.0 Release Notes](https://numpy.org/devdocs/release/2.0.0-notes.html)
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) - Minimum Python versions

---

## File Inventory

### Files to Create

**Build System**:
- `pyproject.toml` - Modern Python packaging configuration
- `meson.build` (root) - Main Meson build file
- `redblackgraph/meson.build` - Package-level build configuration
- `redblackgraph/core/meson.build` - Core C extensions build
- `redblackgraph/sparse/meson.build` - Sparse extensions build
- `redblackgraph/sparse/csgraph/meson.build` - Csgraph Cython extensions
- `redblackgraph/types/meson.build` - Types module build
- `redblackgraph/reference/meson.build` - Reference module build
- `redblackgraph/util/meson.build` - Util module build
- `tools/conv_template.py` - Template preprocessor (if using Option A)

**CI/CD**:
- `.github/workflows/test.yml` - Testing workflow
- `.github/workflows/build.yml` - Wheel building workflow
- `.github/workflows/publish.yml` - PyPI publishing workflow
- `.github/dependabot.yml` - Dependency update automation (optional)

**Testing**:
- `scripts/test_installation.py` - Installation verification script
- `scripts/benchmark.py` - Performance benchmarking (optional)

**Documentation**:
- `MIGRATION.md` - Migration guide for users and developers
- Update `CHANGELOG.md` - Document all changes

### Files to Modify

**Phase 1**:
- `setup.py` - Update version constraints temporarily
- `requirements.txt` - Update dependency versions
- `requirements-dev.txt` - Update development dependencies
- `.travis.yml` - Update Python versions (or create GitHub Actions)
- `README.md` - Update badges and installation instructions
- `.gitignore` - Add Meson build artifacts

**Phase 2+**:
- `MANIFEST.in` - May need updates for Meson
- C extension files - Add NumPy 2.0 API compatibility
- Cython `.pyx` files - Update for NumPy 2.0
- `bin/build-*.sh` scripts - Update for Meson builds
- Docker files - Update build commands

### Files to Remove

**After Phase 2 (Meson migration complete)**:
- `setup.py` (root)
- `redblackgraph/setup.py`
- `redblackgraph/core/setup.py`
- `redblackgraph/sparse/setup.py`
- `redblackgraph/sparse/csgraph/setup.py`
- `setup.cfg` (if no longer needed)

**After Phase 4 (GitHub Actions migration)**:
- `.travis.yml`

**Potentially Deprecated**:
- `requirements-numba.txt` (if Numba support dropped)
- `tools/cythonize.py` (Meson handles Cython natively)

---

## Optional Enhancements

### Code Modernization
**Priority**: Medium  
**Effort**: 1-2 weeks

- **Type Hints**: Add Python 3.10+ type annotations to public APIs
- **Error Messages**: Improve validation and error messages
- **Logging**: Add structured logging for debugging
- **Code Layout**: Consider `src/` layout for cleaner packaging

### Documentation Improvements
**Priority**: High for user adoption  
**Effort**: 1-2 weeks

- Setup Sphinx documentation with API reference
- Integrate Jupyter notebooks with nbsphinx
- Create comprehensive user guide
- Add contributing guide and code of conduct
- Improve README with better examples

### Testing Enhancements
**Priority**: High for quality  
**Effort**: 1-2 weeks

- Increase test coverage to >90%
- Add integration tests for complete workflows
- Add performance benchmarks with tracking
- Consider property-based testing with Hypothesis
- Add regression test suite for critical bugs

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-21 | AI Assistant | Initial migration specification created from upgrade planning documents |

---

## Summary

This migration specification provides a comprehensive roadmap to modernize the RedBlackGraph library. The phased approach allows for controlled risk management while ensuring the library remains compatible with modern Python ecosystem standards.

**Key Takeaways**:
1. **Phase 1** establishes a stable baseline with NumPy 1.26
2. **Phase 2** is the most critical - migrating from numpy.distutils to Meson
3. **Phase 3** upgrades to NumPy 2.0 for long-term viability
4. **Phase 4** modernizes CI/CD infrastructure

**Success depends on**:
- Thorough testing at each phase
- Maintaining stable fallback points
- Clear documentation of changes
- Community communication about breaking changes

**Timeline**: 4-6 weeks conservative, 2.5-3 weeks aggressive

**Next Steps**: Review and approve this specification, then begin Phase 1 implementation.
