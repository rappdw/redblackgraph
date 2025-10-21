# RedBlackGraph Upgrade Checklist

## Phase 1: Initial Upgrade (Python 3.10, NumPy 1.26, SciPy 1.11)

### Code Changes
- [ ] Update `setup.py`:
  - [ ] Change `python_requires='>=3.6'` to `python_requires='>=3.10'`
  - [ ] Change `numpy>=0.14.0` to `numpy>=1.26.0,<2.0.0`
  - [ ] Change `setup_requires` numpy to `numpy>=1.26.0,<2.0.0`
  - [ ] Add `scipy>=1.11.0` to install_requires
  - [ ] Remove Python 3.5-3.9 from classifiers
  - [ ] Add Python 3.10, 3.11, 3.12 to classifiers
  - [ ] Remove `dataclasses;python_version<"3.7"` dependency

- [ ] Update `requirements.txt`:
  - [ ] Remove `dataclasses; python_version < '3.7'` line
  - [ ] Change `numpy` to `numpy>=1.26.0,<2.0.0`
  - [ ] Change `scipy` to `scipy>=1.11.0`

- [ ] Update `requirements-dev.txt`:
  - [ ] Ensure all dev dependencies are compatible

- [ ] Update `.travis.yml`:
  - [ ] Change Python versions from `["3.6", "3.7", "3.8"]` to `["3.10", "3.11", "3.12"]`
  - [ ] Update dist from `xenial` to `jammy` or `focal`
  
- [ ] Update `README.md`:
  - [ ] Update Python version badges
  - [ ] Update installation instructions if needed

### Testing
- [ ] Install dependencies in clean virtual environment
- [ ] Run `python setup.py build_ext --inplace`
- [ ] Run full test suite: `pytest tests/`
- [ ] Check for deprecation warnings
- [ ] Test key functionality manually:
  - [ ] Matrix operations
  - [ ] Sparse matrix operations
  - [ ] Csgraph operations
  - [ ] Command line tool (`rbg`)
- [ ] Test on multiple Python versions (3.10, 3.11, 3.12)
- [ ] Test wheel building: `python setup.py bdist_wheel`

### Documentation
- [ ] Update changelog
- [ ] Update any version-specific documentation
- [ ] Tag release

## Phase 2: Migrate to Meson Build System

### Setup
- [ ] Study Meson documentation
- [ ] Review SciPy's meson.build files for examples
- [ ] Install Meson and ninja: `pip install meson ninja meson-python`

### Create Build Files
- [ ] Create `pyproject.toml`:
  - [ ] Define build-system section
  - [ ] Define project metadata
  - [ ] Specify dependencies
  
- [ ] Create `meson.build` (root):
  - [ ] Project declaration
  - [ ] Python module detection
  - [ ] Compiler configuration
  - [ ] Add subdirectories
  
- [ ] Create `redblackgraph/meson.build`:
  - [ ] Python sources installation
  - [ ] Subpackages
  
- [ ] Create `redblackgraph/core/meson.build`:
  - [ ] C extension configuration
  - [ ] Handle .c.src templating
  - [ ] Link NumPy
  
- [ ] Create `redblackgraph/sparse/meson.build`:
  - [ ] Sparsetools extension
  - [ ] Subpackage csgraph
  
- [ ] Create `redblackgraph/sparse/csgraph/meson.build`:
  - [ ] All Cython extensions
  - [ ] NumPy dependencies

### Handle Special Cases
- [ ] Convert or replace `.c.src` templating:
  - [ ] Option 1: Create Python script to preprocess
  - [ ] Option 2: Convert to regular C with templates
  - [ ] Option 3: Convert to Cython
  
- [ ] Update `generate_sparsetools.py` if needed
  
- [ ] Update or remove `cythonize.py` script
  - [ ] Meson handles Cython natively

### Testing
- [ ] Clean build: `rm -rf build/`
- [ ] Test build: `python -m build`
- [ ] Test install: `pip install -e .`
- [ ] Run all tests
- [ ] Test wheel creation
- [ ] Test on clean environment
- [ ] Test on multiple platforms:
  - [ ] Linux
  - [ ] macOS
  - [ ] Windows (if applicable)

### Cleanup
- [ ] Remove old `setup.py` files:
  - [ ] Root `setup.py`
  - [ ] `redblackgraph/setup.py`
  - [ ] `redblackgraph/core/setup.py`
  - [ ] `redblackgraph/sparse/setup.py`
  - [ ] `redblackgraph/sparse/csgraph/setup.py`
  
- [ ] Update `setup.cfg` or remove if not needed
  
- [ ] Update `MANIFEST.in` if needed
  
- [ ] Update `.gitignore` for Meson build artifacts

### Documentation
- [ ] Update build instructions in README
- [ ] Update developer documentation
- [ ] Update wheel building scripts
- [ ] Update Docker files if needed

## Phase 3: Upgrade to NumPy 2.0

### Preparation
- [ ] Read NumPy 2.0 migration guide thoroughly
- [ ] Check C API changes relevant to extensions
- [ ] Ensure Cython version supports NumPy 2.0 (Cython >= 3.0)

### Code Updates
- [ ] Update `pyproject.toml`:
  - [ ] Change numpy constraint to `>=2.0.0`
  
- [ ] Review and update C extensions:
  - [ ] Check `redblackgraphmodule.c`
  - [ ] Check `.c.src` files (if still using)
  - [ ] Update NumPy C API calls
  - [ ] Update array creation
  - [ ] Check dtype handling
  
- [ ] Review and update Cython extensions:
  - [ ] All `.pyx` files in `sparse/csgraph/`
  - [ ] Update numpy cimports
  - [ ] Check array declarations
  
- [ ] Update Python code:
  - [ ] Check for deprecated NumPy functions
  - [ ] Update dtype specifications if needed
  - [ ] Check array creation patterns

### Testing
- [ ] Clean rebuild with NumPy 2.0
- [ ] Run full test suite
- [ ] Check for runtime warnings
- [ ] Performance testing:
  - [ ] Benchmark critical operations
  - [ ] Compare with NumPy 1.26 performance
  - [ ] Document any changes
  
- [ ] Compatibility testing:
  - [ ] Test with minimum supported SciPy version
  - [ ] Test with latest SciPy version
  
- [ ] Memory testing:
  - [ ] Check for memory leaks
  - [ ] Validate reference counting

### Documentation
- [ ] Update installation requirements
- [ ] Note any breaking changes
- [ ] Update performance documentation if needed
- [ ] Tag major version release

## Phase 4: CI/CD Modernization

### GitHub Actions Migration
- [ ] Create `.github/workflows/test.yml`:
  - [ ] Test matrix (Python 3.10, 3.11, 3.12)
  - [ ] Multiple OS (Ubuntu, macOS)
  - [ ] Install dependencies
  - [ ] Run tests
  - [ ] Upload coverage
  
- [ ] Create `.github/workflows/build.yml`:
  - [ ] Build wheels
  - [ ] Test wheels
  - [ ] Upload artifacts
  
- [ ] Create `.github/workflows/publish.yml`:
  - [ ] Publish to PyPI on release
  
- [ ] Remove `.travis.yml` once GitHub Actions working

### Additional CI Improvements
- [ ] Add pre-commit hooks:
  - [ ] Code formatting (black, ruff)
  - [ ] Linting (pylint, ruff)
  - [ ] Type checking (mypy) if adding type hints
  
- [ ] Update codecov integration
  
- [ ] Add dependabot for dependency updates

## Optional Enhancements

### Code Modernization
- [ ] Add type hints (Python 3.10+ syntax):
  - [ ] Core modules
  - [ ] Public APIs
  - [ ] Tests
  
- [ ] Improve error messages
  
- [ ] Add logging where appropriate
  
- [ ] Consider src/ layout for package

### Documentation
- [ ] Setup Sphinx documentation
  
- [ ] Integrate notebooks with nbsphinx
  
- [ ] Add API reference
  
- [ ] Add contributing guide
  
- [ ] Add code of conduct

### Testing
- [ ] Increase test coverage
  
- [ ] Add integration tests
  
- [ ] Add performance benchmarks
  
- [ ] Add property-based testing (hypothesis)

## Verification Steps (After Each Phase)

### Build Verification
```bash
# Clean environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install and test
pip install -e ".[test]"
pytest tests/ -v --cov=redblackgraph

# Test wheel
pip install build
python -m build
pip install dist/RedBlackGraph-*.whl
python -c "import redblackgraph; print(redblackgraph.__version__)"
```

### Import Test
```python
# Verify all imports work
import redblackgraph
from redblackgraph.core import redblack
from redblackgraph.sparse import rb_matrix
from redblackgraph.sparse.csgraph import (
    transitive_closure,
    all_pairs_avos_path,
    components,
)
print("All imports successful!")
```

### Functionality Test
```bash
# Test command line tool
rbgcf --help

# Run notebooks (if applicable)
jupyter notebook notebooks/
```

## Rollback Procedures

### If Phase 1 Fails
```bash
git checkout main
# Revert changes
# Keep Python 3.6+ support
# Stay on NumPy < 1.26
```

### If Phase 2 Fails
```bash
git checkout phase1-stable
# Keep using numpy.distutils
# Stay on NumPy 1.26
# Can maintain this state for extended period
```

### If Phase 3 Fails
```bash
git checkout phase2-stable
# Keep using NumPy 1.26
# Meson build system still works
# Can maintain this state for extended period
```

## Success Criteria

### Phase 1 Complete
- ✅ All tests pass with Python 3.10, 3.11, 3.12
- ✅ All tests pass with NumPy 1.26.x
- ✅ All tests pass with SciPy 1.11+
- ✅ No deprecation warnings (or documented)
- ✅ Wheels build successfully
- ✅ Package installable via pip

### Phase 2 Complete
- ✅ Meson build successful
- ✅ All extensions compile
- ✅ All tests pass
- ✅ Wheels build with Meson
- ✅ No numpy.distutils dependencies
- ✅ Build works on multiple platforms

### Phase 3 Complete
- ✅ All tests pass with NumPy 2.0+
- ✅ No runtime warnings
- ✅ Performance comparable or better
- ✅ Memory usage acceptable
- ✅ Documentation updated

## Notes and Issues

### Known Issues
- [ ] Document any known issues here
- [ ] Link to GitHub issues

### Questions/Decisions
- [ ] Which NumPy 2.0 version to target?
- [ ] Drop Python 3.9 support or keep it?
- [ ] Keep numba support? (requirements-numba.txt is old)

### Resources Used
- [ ] List helpful resources
- [ ] Link to examples followed
- [ ] Note any deviations from plan
