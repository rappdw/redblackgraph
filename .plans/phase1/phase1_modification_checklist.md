# Phase 1 Modification Checklist

**Date**: 2025-10-21  
**Purpose**: Tracking checklist for all Phase 1 code modifications  
**Sprint**: Phase 1, Sprints 2-4

---

## Sprint 2: Dependency Updates

**Owner**: Build Engineer  
**Estimated Effort**: 3-4 days

### setup.py Updates (Root)

- [ ] Update Python version requirement: `python_requires='>=3.10'`
- [ ] Remove dataclasses conditional: Delete `'dataclasses;python_version<"3.7"'`
- [ ] Update NumPy constraint: `'numpy>=1.26.0,<1.27'`
- [ ] Update SciPy constraint: `'scipy>=1.11.0'`
- [ ] Add Cython constraint: `'cython>=3.0'` in setup_requires
- [ ] Add temporary setuptools constraint: `'setuptools<60.0'` (for numpy.distutils)
- [ ] Update Python classifiers:
  - [ ] Remove: 'Programming Language :: Python :: 3.5'
  - [ ] Remove: 'Programming Language :: Python :: 3.6'
  - [ ] Remove: 'Programming Language :: Python :: 3.7'
  - [ ] Remove: 'Programming Language :: Python :: 3.8'
  - [ ] Remove: 'Programming Language :: Python :: 3.9'
  - [ ] Keep: 'Programming Language :: Python :: 3.10'
  - [ ] Add: 'Programming Language :: Python :: 3.11'
  - [ ] Add: 'Programming Language :: Python :: 3.12'

### requirements.txt Updates

- [ ] Remove dataclasses line: Delete `dataclasses; python_version < '3.7'`
- [ ] Update NumPy: `numpy>=1.26.0,<1.27`
- [ ] Update SciPy: `scipy>=1.11.0`
- [ ] Verify XlsxWriter compatibility with Python 3.10+
- [ ] Verify fs-crawler>=0.3.2 compatibility with Python 3.10+

### requirements-dev.txt / requirements-test.txt

- [ ] Update pytest if needed
- [ ] Update pytest-cov if needed
- [ ] Update pylint if needed
- [ ] Update Cython: `cython>=3.0`

### .travis.yml Updates

- [ ] Update Python versions:
  - [ ] Remove: "3.6"
  - [ ] Remove: "3.7"
  - [ ] Remove: "3.8"
  - [ ] Add: "3.10"
  - [ ] Add: "3.11"
  - [ ] Add: "3.12"
- [ ] Update dist if needed (xenial → newer)
- [ ] Verify codecov integration still works

### README.md Updates

- [ ] Update Python version badges:
  - [ ] Remove Python 3.5, 3.6, 3.7, 3.8, 3.9 badges
  - [ ] Keep Python 3.10 badge
  - [ ] Add Python 3.11 badge
  - [ ] Add Python 3.12 badge

### Versioneer Check

- [ ] Identify current versioneer version
- [ ] Check for latest versioneer release
- [ ] Update versioneer if newer version available
- [ ] Test version generation works

### setup.py Files Review

- [ ] Review `./setup.py` - ✅ Primary updates here
- [ ] Review `./redblackgraph/setup.py` - No changes needed
- [ ] Review `./redblackgraph/core/setup.py` - No changes needed
- [ ] Review `./redblackgraph/sparse/setup.py` - No changes needed
- [ ] Review `./redblackgraph/sparse/csgraph/setup.py` - No changes needed

### Build Verification

- [ ] Build succeeds on Python 3.10
- [ ] Build succeeds on Python 3.11
- [ ] Build succeeds on Python 3.12
- [ ] Wheel creation succeeds for all versions
- [ ] Installation from wheel succeeds

### Sprint 2 Acceptance Criteria

- [ ] All dependency constraints updated
- [ ] All Python version references updated
- [ ] Builds succeed without errors on Python 3.10, 3.11, 3.12
- [ ] No deprecation warnings during build
- [ ] Travis CI passes on all three Python versions

---

## Sprint 3: Code Modernization

**Owner**: Software Engineers  
**Estimated Effort**: 3-4 days

### Dataclasses Migration

- [ ] Verify `redblackgraph/types/transitive_closure.py` imports from stdlib
- [ ] Verify `redblackgraph/types/relationship.py` imports from stdlib
- [ ] Verify `redblackgraph/types/ordering.py` imports from stdlib
- [ ] Test all dataclass functionality works identically
- [ ] Remove any references to dataclasses backport

### NumPy Deprecated Types (Python Code)

**Status**: ✅ No deprecated types found in scan

- [ ] Confirm no `np.int` usage (already verified)
- [ ] Confirm no `np.float` usage (already verified)
- [ ] Confirm no `np.bool` usage (already verified)
- [ ] Check for `numpy.random.RandomState` vs new Generator API
- [ ] Update any found instances to use explicit dtypes

### C Extension Updates (High Priority)

#### NPY_NO_EXPORT Macro Fix

- [ ] Audit all uses of `NPY_NO_EXPORT` in .c.src files
- [ ] **File: rbg_math.c.src**
  - [ ] Replace `NPY_NO_EXPORT` with `static` or remove
  - [ ] Test compilation
- [ ] **File: redblack.c.src**
  - [ ] Replace `NPY_NO_EXPORT` with `static` or remove
  - [ ] Test compilation
- [ ] **File: relational_composition.c.src**
  - [ ] Replace `NPY_NO_EXPORT` with `static` or remove
  - [ ] Test compilation
- [ ] **File: warshall.c.src**
  - [ ] Replace `NPY_NO_EXPORT` with `static` or remove
  - [ ] Test compilation

#### NumPy C API Compatibility

- [ ] Review NumPy 1.26 C API changelog
- [ ] Remove unnecessary `npy_3kcompat.h` includes (Python 2 compat)
- [ ] Verify `NPY_NO_DEPRECATED_API` usage is correct
- [ ] Test all C extensions compile with NumPy 1.26
- [ ] Run tests to verify C extension functionality

#### einsum Internal API

- [ ] Identify exact einsum internal API usage in relational_composition.c.src
- [ ] Check if NumPy 1.26 changed einsum internals
- [ ] Test einsum functionality extensively
- [ ] Consider using public NumPy API if internals changed
- [ ] Document any workarounds needed

### Cython Extension Updates

- [ ] **File: _components.pyx** - Test with Cython 3.1.5
- [ ] **File: _ordering.pyx** - Test with Cython 3.1.5
- [ ] **File: _permutation.pyx** - Test with Cython 3.1.5
- [ ] **File: _rbg_math.pyx** - Test with Cython 3.1.5
- [ ] **File: _relational_composition.pyx** - Test with Cython 3.1.5
- [ ] **File: _shortest_path.pyx** - Test with Cython 3.1.5
- [ ] **File: _tools.pyx** - Test with Cython 3.1.5
- [ ] Fix any Cython 3.x deprecation warnings
- [ ] Verify all Cython extensions load successfully

### Code Quality

- [ ] No compiler warnings
- [ ] No deprecation warnings from NumPy
- [ ] No deprecation warnings from Python
- [ ] Linting passes (pylint)
- [ ] Code follows existing style conventions

### Sprint 3 Acceptance Criteria

- [ ] All C extensions compile without errors
- [ ] All Cython extensions compile without errors
- [ ] No `NPY_NO_EXPORT` usage remaining
- [ ] No deprecated NumPy C API usage
- [ ] All extensions load and import successfully
- [ ] Unit tests pass (even if not all)

---

## Sprint 4: Testing & Validation

**Owner**: QA Engineers  
**Estimated Effort**: 4-5 days

### Test Execution

#### Python 3.10 Testing
- [ ] Full test suite passes on Python 3.10
- [ ] No test failures
- [ ] No test errors
- [ ] Test duration acceptable (< 10 min)
- [ ] Code coverage meets/exceeds baseline

#### Python 3.11 Testing
- [ ] Full test suite passes on Python 3.11
- [ ] No test failures
- [ ] No test errors
- [ ] Test duration acceptable (< 10 min)
- [ ] Code coverage meets/exceeds baseline

#### Python 3.12 Testing
- [ ] Full test suite passes on Python 3.12
- [ ] No test failures
- [ ] No test errors
- [ ] Test duration acceptable (< 10 min)
- [ ] Code coverage meets/exceeds baseline

### Platform Testing

#### Linux (aarch64)
- [ ] Build succeeds
- [ ] All tests pass on Python 3.10
- [ ] All tests pass on Python 3.11
- [ ] All tests pass on Python 3.12
- [ ] Wheel creation succeeds

#### macOS (if available)
- [ ] Build succeeds
- [ ] All tests pass on Python 3.10
- [ ] All tests pass on Python 3.11
- [ ] All tests pass on Python 3.12
- [ ] Wheel creation succeeds

### CI/CD Validation

- [ ] Travis CI builds pass on all Python versions
- [ ] Code coverage uploaded to codecov successfully
- [ ] Coverage percentage acceptable
- [ ] No CI failures or warnings

### Warning Checks

- [ ] No DeprecationWarnings from NumPy
- [ ] No DeprecationWarnings from SciPy
- [ ] No DeprecationWarnings from Python
- [ ] No FutureWarnings
- [ ] No compiler warnings during build

### Build Artifacts

- [ ] Wheels build successfully for Python 3.10
- [ ] Wheels build successfully for Python 3.11
- [ ] Wheels build successfully for Python 3.12
- [ ] Source distribution (sdist) builds successfully
- [ ] Artifacts install in clean environments
- [ ] Entry points work correctly (scripts/rbg)

### Integration Testing

- [ ] Import tests: `import redblackgraph` succeeds
- [ ] Core module: `import redblackgraph.core` succeeds
- [ ] Sparse module: `import redblackgraph.sparse` succeeds
- [ ] CSGraph module: `import redblackgraph.sparse.csgraph` succeeds
- [ ] Reference module: `import redblackgraph.reference` succeeds
- [ ] Types module: `import redblackgraph.types` succeeds
- [ ] Util module: `import redblackgraph.util` succeeds

### End-to-End Testing

- [ ] Run example from README
- [ ] Verify rbgcf command works
- [ ] Test with fs-crawler integration
- [ ] Verify Excel output generation works

### Docker Testing

- [ ] Update Dockerfile(s) to use Python 3.10+
- [ ] Docker images build successfully
- [ ] Tests pass inside Docker containers
- [ ] Document Docker image usage

### Jupyter Notebook Updates

- [ ] Identify all notebooks in `notebooks/` directory
- [ ] Update Python version requirements in notebooks
- [ ] Test notebooks execute without errors
- [ ] Update any deprecated API usage in notebooks
- [ ] Verify visualizations render correctly

### Documentation Review

- [ ] README.md reflects correct Python versions
- [ ] Installation instructions work
- [ ] Development setup instructions work
- [ ] API examples still valid
- [ ] Links are not broken

### Sprint 4 Acceptance Criteria

- [ ] 100% test pass rate on all Python versions (3.10, 3.11, 3.12)
- [ ] Builds succeed on Linux and macOS
- [ ] Travis CI passes
- [ ] No deprecation warnings
- [ ] Docker images working
- [ ] Jupyter notebooks updated
- [ ] Documentation updated
- [ ] Ready for merge to master

---

## Final Phase 1 Checklist

### Code Quality Gates

- [ ] All tests passing across Python 3.10, 3.11, 3.12
- [ ] No critical deprecation warnings
- [ ] Code coverage maintained or improved
- [ ] Linting passes
- [ ] No compiler warnings

### Build & Distribution

- [ ] Builds successful on Linux and macOS
- [ ] Wheels build for all Python versions
- [ ] Installation tested in clean environments
- [ ] Entry points functional
- [ ] Source distribution works

### Documentation

- [ ] README.md updated
- [ ] Development setup guide complete
- [ ] All .plans/ documentation complete
- [ ] Jupyter notebooks updated
- [ ] CHANGELOG prepared (if exists)

### Version Control

- [ ] All changes committed to migration branch
- [ ] Sprint completion tags created
- [ ] Branch ready for merge to master

### Stakeholder Approval

- [ ] Tech lead sign-off (self-approval)
- [ ] QA validation complete
- [ ] Project owner approval (Daniel Rapp)

### Release Preparation

- [ ] Version bumped to 2.0.0
- [ ] Git tag prepared: v2.0.0
- [ ] Release notes drafted
- [ ] PyPI release strategy defined

---

## Post-Merge Actions

**After merging migration branch to master:**

- [ ] Merge migration to master
- [ ] Tag release: `git tag -a v2.0.0 -m "Phase 1: Python 3.10+ migration"`
- [ ] Push to GitHub: `git push origin master --tags`
- [ ] Build and upload wheels to PyPI (if manual process)
- [ ] Verify PyPI page shows correct Python versions
- [ ] Update GitHub releases page
- [ ] Announce migration completion (if applicable)

---

## Notes

- **Sprint 2** must complete before Sprint 3 can begin (dependency on build working)
- **Sprint 3** must complete before Sprint 4 can begin (dependency on code working)
- **Blocking issues** should be escalated immediately to project owner
- **Optional items** in Sprint 4 (like Docker) can be deferred if timeline pressured

---

## Risk Mitigation

If any checklist item cannot be completed:

1. **Document the blocker** in `.plans/sprint_X_blockers.md`
2. **Assess impact** - Can we proceed or must we stop?
3. **Escalate** to project owner if blocking
4. **Create workaround** if possible
5. **Update timeline** if needed

---

**Document Status**: Ready for Use  
**Last Updated**: 2025-10-21  
**Owner**: Tech Lead / Project Manager
