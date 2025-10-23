# Phase 2 Specification: Meson Build System Migration

**Document Version**: 1.0  
**Date**: 2025-10-21  
**Status**: Draft  
**Branch**: `phase2-meson`  
**Based On**: `master` (v0.4.0-phase1)

---

## Executive Summary

Migrate RedBlackGraph from `numpy.distutils` to Meson build system to enable Python 3.12+ support and align with NumPy/SciPy ecosystem standards.

**Primary Goal**: Python 3.12 support  
**Secondary Goals**: Faster builds, modern tooling, future-proof infrastructure  
**Timeline**: 5-11 days  
**Risk Level**: Medium (well-documented migration path)

---

## 1. Background & Motivation

### 1.1 Problem Statement

**Current Blocker**: Python 3.12 removed `distutils` from stdlib (PEP 632)

RedBlackGraph currently uses `numpy.distutils` extensively:
- Lines 12, 36, 158 in `/setup.py`
- Lines 5, 17 in `/redblackgraph/setup.py`
- Lines 7, 27 in `/redblackgraph/core/setup.py`
- Lines 9, 42 in `/redblackgraph/sparse/setup.py`
- Line 3 in `/redblackgraph/sparse/csgraph/setup.py`

**Timeline**:
- NumPy 1.26: `numpy.distutils` deprecated
- NumPy 2.0: `numpy.distutils` removed
- Python 3.12: `distutils` removed from stdlib

### 1.2 Solution: Meson Build System

**Why Meson**:
- Official NumPy/SciPy recommendation
- Used by NumPy, SciPy, scikit-image
- Faster builds (parallel compilation)
- Better dependency tracking
- Modern, actively maintained
- Excellent Python integration via meson-python

---

## 2. Objectives & Success Criteria

### 2.1 Primary Objectives

1. **Python 3.12 Support**: Build and test on Python 3.12.x
2. **Backward Compatibility**: Maintain Python 3.10, 3.11 support
3. **Feature Parity**: All build features preserved
4. **Test Success**: 117/117 tests passing on all Python versions
5. **Zero Dependencies on numpy.distutils**: Complete removal

### 2.2 Success Criteria

Phase 2 is **COMPLETE** when:
- ✅ All C/C++ extensions build with Meson
- ✅ All Cython modules compile with Meson
- ✅ Python 3.10: 117/117 tests pass
- ✅ Python 3.11: 117/117 tests pass
- ✅ Python 3.12: 117/117 tests pass
- ✅ Build time ≤ current build time (ideally faster)
- ✅ No `numpy.distutils` imports remain
- ✅ CI/CD updated and passing
- ✅ Documentation updated
- ✅ Developer setup documented

### 2.3 Out of Scope

- NumPy 2.0 migration (separate phase)
- Performance optimization beyond build system
- New features or functionality
- Refactoring existing code (unless required for Meson)

---

## 3. Technical Requirements

### 3.1 Build System Requirements

| Component | Current | Target |
|-----------|---------|--------|
| **Build System** | numpy.distutils | Meson + meson-python |
| **Build Backend** | setuptools | meson-python (PEP 517) |
| **Configuration** | setup.py | pyproject.toml + meson.build |
| **Template Processing** | .src files (NumPy style) | Meson templating |
| **Cython Integration** | numpy.distutils | Meson cython module |
| **Symbol Visibility** | Version scripts (GNU ld) | Meson visibility |

### 3.2 Python Version Support

- **Python 3.10**: Full support (baseline)
- **Python 3.11**: Full support (baseline)
- **Python 3.12**: NEW - Primary goal

### 3.3 Dependency Requirements

**Build Dependencies**:
```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "meson>=1.2.0",
    "ninja",
    "Cython>=3.0.0",
    "numpy>=1.26.0"
]
build-backend = "mesonpy"
```

**Runtime Dependencies** (unchanged):
```toml
dependencies = [
    "numpy>=1.26.0,<2.0",
    "scipy>=1.11.0",
    "XlsxWriter",
    "fs-crawler>=0.3.2"
]
```

### 3.4 Extension Requirements

**C Extensions** (7 total):
1. `redblackgraph.core._redblack`
2. `redblackgraph.core._warshall`
3. `redblackgraph.core._relational_composition`
4. `redblackgraph.sparse._sparsetools`
5. `redblackgraph.sparse.csgraph._traversal`
6. `redblackgraph.sparse.csgraph._flow`
7. `redblackgraph.sparse.csgraph._matching`

**Cython Modules** (7 total):
- All in `sparse/` and `core/` directories
- Must compile with Cython 3.0+

**Template Files** (.src):
- `rbg_math.c.src`
- `rbg_math.h.src`
- `redblack.c.src`
- `warshall.c.src`
- `relational_composition.c.src`
- And others - must be converted to Meson templating

---

## 4. Sprint Breakdown

### Sprint 1: Meson Setup & Learning (2-3 days)

**Objectives**:
- Install and configure Meson toolchain
- Understand Meson build system
- Create minimal working build
- Validate approach

**Tasks**:

**1.1 Environment Setup** (2 hours)
- [ ] Install Meson: `pip install meson>=1.2.0`
- [ ] Install ninja: `pip install ninja`
- [ ] Install meson-python: `pip install meson-python>=0.15.0`
- [ ] Verify installations: `meson --version`, `ninja --version`

**1.2 Research & Learning** (4-6 hours)
- [ ] Clone NumPy reference: `git clone https://github.com/numpy/numpy.git`
- [ ] Clone SciPy reference: `git clone https://github.com/scipy/scipy.git`
- [ ] Study NumPy's root `meson.build`
- [ ] Study SciPy's `scipy/meson.build`
- [ ] Review meson-python documentation
- [ ] Review Meson Python module docs

**1.3 Create Base Configuration** (4 hours)
- [ ] Create `pyproject.toml` with build-system section
- [ ] Create root `meson.build` file
- [ ] Configure project metadata
- [ ] Set up Python detection
- [ ] Configure NumPy include paths
- [ ] Add `redblackgraph` subdir

**1.4 Test Minimal Build** (2-4 hours)
- [ ] Attempt build: `python -m build --no-isolation`
- [ ] Debug any configuration issues
- [ ] Verify meson setup works: `meson setup builddir`
- [ ] Document findings and issues

**Deliverables**:
- Working `pyproject.toml`
- Root `meson.build` (minimal)
- Build test results
- Sprint 1 completion summary

---

### Sprint 2: Core Extension Migration (1-2 days)

**Objectives**:
- Migrate all core C extensions to Meson
- Handle `.src` template conversion
- Build and test core functionality

**Tasks**:

**2.1 Core Configuration** (2 hours)
- [ ] Create `redblackgraph/core/meson.build`
- [ ] Configure core subproject structure
- [ ] Set up include directories
- [ ] Configure compiler flags

**2.2 Template Conversion** (4-6 hours)
- [ ] Study NumPy template processing (`.src` → Meson)
- [ ] Convert `rbg_math.c.src` to Meson templating
- [ ] Convert `rbg_math.h.src` to Meson templating
- [ ] Convert `redblack.c.src` to Meson templating
- [ ] Convert `warshall.c.src` to Meson templating
- [ ] Convert `relational_composition.c.src` to Meson templating
- [ ] Test template expansion

**2.3 Extension Building** (2-4 hours)
- [ ] Configure `_redblack` extension
- [ ] Configure `_warshall` extension
- [ ] Configure `_relational_composition` extension
- [ ] Set up source lists
- [ ] Configure dependencies (NumPy C API)
- [ ] Build extensions: `meson compile -C builddir`

**2.4 Testing** (2 hours)
- [ ] Import test: `python -c "import redblackgraph.core._redblack"`
- [ ] Run core tests: `pytest tests/core/`
- [ ] Debug any import or runtime issues
- [ ] Verify functionality

**Deliverables**:
- `redblackgraph/core/meson.build`
- Converted template files
- Working core extensions
- Core test pass confirmation

---

### Sprint 3: Sparse Extension Migration (1-2 days)

**Objectives**:
- Migrate sparse C/C++ extensions
- Migrate Cython modules
- Implement symbol visibility

**Tasks**:

**3.1 Sparse Configuration** (2 hours)
- [ ] Create `redblackgraph/sparse/meson.build`
- [ ] Create `redblackgraph/sparse/csgraph/meson.build`
- [ ] Configure subproject structure
- [ ] Set up C++ compilation

**3.2 Cython Integration** (3-4 hours)
- [ ] Configure Cython module detection
- [ ] Set up Cython compilation rules
- [ ] Migrate Cython extensions (7 modules)
- [ ] Configure Cython include paths
- [ ] Build Cython extensions

**3.3 C++ Sparsetools** (2-3 hours)
- [ ] Configure C++ compiler
- [ ] Set up `sparsetools/` directory
- [ ] Configure `_sparsetools` extension
- [ ] Handle C++ template headers
- [ ] Build sparse extensions

**3.4 Symbol Visibility** (1-2 hours)
- [ ] Implement version script equivalent
- [ ] Configure symbol visibility on GNU ld
- [ ] Test symbol export/hiding
- [ ] Verify no symbol conflicts

**3.5 Testing** (2 hours)
- [ ] Import test: `python -c "import redblackgraph.sparse"`
- [ ] Run sparse tests: `pytest tests/sparse/`
- [ ] Run rb_matrix tests specifically
- [ ] Verify AVOS multiplication works

**Deliverables**:
- `redblackgraph/sparse/meson.build`
- `redblackgraph/sparse/csgraph/meson.build`
- Working Cython modules
- Working sparse extensions
- Sparse test pass confirmation

---

### Sprint 4: Python 3.12 Testing (1 day)

**Objectives**:
- Create Python 3.12 environment
- Build and test on Python 3.12
- Fix any Python 3.12-specific issues

**Tasks**:

**4.1 Environment Setup** (30 minutes)
- [ ] Create Python 3.12 venv: `uv venv .venv-3.12 --python 3.12`
- [ ] Install build tools
- [ ] Install test dependencies

**4.2 Build on Python 3.12** (1-2 hours)
- [ ] Clean build: `rm -rf build/`
- [ ] Build with meson-python
- [ ] Install in development mode
- [ ] Verify all extensions load

**4.3 Test Suite Execution** (2-3 hours)
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Target: 117/117 tests passing
- [ ] Document any failures
- [ ] Fix Python 3.12-specific issues (if any)

**4.4 Cross-Version Verification** (1-2 hours)
- [ ] Test on Python 3.10: `pytest tests/`
- [ ] Test on Python 3.11: `pytest tests/`
- [ ] Test on Python 3.12: `pytest tests/`
- [ ] Verify all pass 117/117

**4.5 Performance Verification** (1 hour)
- [ ] Measure build time (compare to Phase 1)
- [ ] Measure import time
- [ ] Run performance-sensitive tests
- [ ] Document any regressions

**Deliverables**:
- Python 3.12 test results
- Cross-version test matrix
- Performance comparison
- Issue list (if any)

---

### Sprint 5: CI/CD & Documentation (1 day)

**Objectives**:
- Update CI/CD configuration
- Update all documentation
- Prepare for release

**Tasks**:

**5.1 CI/CD Update** (2-3 hours)
- [ ] Update `.travis.yml` or migrate to GitHub Actions
- [ ] Add Python 3.12 to test matrix
- [ ] Configure Meson in CI
- [ ] Test CI build
- [ ] Verify CI tests pass

**5.2 Documentation Updates** (2-3 hours)
- [ ] Update `README.md` (Python 3.12 support, build instructions)
- [ ] Create `BUILDING.md` (detailed Meson build guide)
- [ ] Update `setup.py` deprecation notice (if keeping for compat)
- [ ] Update developer documentation
- [ ] Document Meson workflow

**5.3 Migration Guide** (1-2 hours)
- [ ] Create migration guide for contributors
- [ ] Document differences from numpy.distutils
- [ ] Provide troubleshooting tips
- [ ] Link to Meson resources

**5.4 Release Preparation** (1 hour)
- [ ] Update version number
- [ ] Create CHANGELOG entry
- [ ] Update classifiers in pyproject.toml
- [ ] Create Phase 2 completion summary

**Deliverables**:
- Updated CI/CD configuration
- `BUILDING.md`
- Updated `README.md`
- Migration guide
- Phase 2 completion summary

---

## 5. File Structure Changes

### 5.1 Files to Create

```
redblackgraph/
├── pyproject.toml           # NEW - PEP 517/518 build config
├── meson.build             # NEW - Root build config
├── meson.options           # NEW - Build options (optional)
├── BUILDING.md             # NEW - Build documentation
├── redblackgraph/
│   ├── meson.build         # NEW - Package build config
│   ├── core/
│   │   └── meson.build     # NEW - Core extensions
│   └── sparse/
│       ├── meson.build     # NEW - Sparse extensions
│       └── csgraph/
│           └── meson.build # NEW - Csgraph extensions
```

### 5.2 Files to Modify

```
redblackgraph/
├── setup.py                # MODIFY or DEPRECATE
├── README.md               # UPDATE - Build instructions
├── .travis.yml             # UPDATE - CI config
└── .github/                # UPDATE - If using GitHub Actions
```

### 5.3 Files to Potentially Remove (Post-Migration)

```
redblackgraph/
├── setup.py                # Can remove after deprecation period
├── redblackgraph/setup.py  # Remove
├── redblackgraph/core/setup.py     # Remove
├── redblackgraph/sparse/setup.py   # Remove
└── redblackgraph/sparse/csgraph/setup.py  # Remove
```

---

## 6. Testing Requirements

### 6.1 Test Matrix

| Python | NumPy | SciPy | Tests | Status |
|--------|-------|-------|-------|--------|
| 3.10.19 | 1.26.4 | 1.15+ | 117/117 | Must Pass |
| 3.11.14 | 1.26.4 | 1.16+ | 117/117 | Must Pass |
| 3.12.x | 1.26.4 | 1.16+ | 117/117 | Must Pass |

### 6.2 Test Categories

**Unit Tests** (117 total):
- Core module tests
- Sparse matrix tests
- Graph algorithm tests
- AVOS operation tests
- Edge case tests

**Integration Tests**:
- Full import test
- Cross-module interaction
- Example notebooks (if applicable)

**Build Tests**:
- Clean build from scratch
- Incremental rebuild
- Cross-platform (if applicable)

**Performance Tests**:
- Build time measurement
- Import time measurement
- Runtime performance spot checks

### 6.3 Regression Tests

Must verify no regression in:
- rb_matrix AVOS multiplication
- All sparse operations
- All graph algorithms
- NumPy/SciPy integration
- Type handling (signed/unsigned integers)

---

## 7. Risk Assessment & Mitigation

### 7.1 High-Risk Areas

**Risk 1: Template Conversion Complexity**
- **Impact**: High (affects multiple files)
- **Probability**: Medium
- **Mitigation**: Study NumPy examples, test incrementally
- **Fallback**: Manual template expansion as last resort

**Risk 2: Symbol Visibility Issues**
- **Impact**: Medium (symbol conflicts)
- **Probability**: Low (Meson handles this well)
- **Mitigation**: Test on Linux first, verify no conflicts

**Risk 3: Python 3.12-Specific Issues**
- **Impact**: High (blocks Phase 2 goal)
- **Probability**: Low (3.12 is stable)
- **Mitigation**: Test early, separate from build system changes

### 7.2 Medium-Risk Areas

**Risk 4: Build Time Regression**
- **Impact**: Medium (developer experience)
- **Probability**: Low (Meson usually faster)
- **Mitigation**: Measure and optimize if needed

**Risk 5: CI/CD Migration**
- **Impact**: Medium (deployment issues)
- **Probability**: Low
- **Mitigation**: Test locally first, parallel CI setup

### 7.3 Mitigation Strategies

1. **Incremental Migration**: Build each component separately
2. **Frequent Testing**: Test after each major change
3. **Rollback Plan**: Keep phase2-meson branch separate
4. **Reference Implementation**: Use NumPy/SciPy as examples
5. **Community Support**: Meson Discord, NumPy mailing list

---

## 8. Timeline & Milestones

### 8.1 Conservative Timeline (11 days)

| Week | Days | Sprints | Milestones |
|------|------|---------|------------|
| 1 | Mon-Wed | Sprint 1 | Meson setup complete |
| 1 | Thu-Fri | Sprint 2 | Core extensions working |
| 2 | Mon-Tue | Sprint 3 | Sparse extensions working |
| 2 | Wed | Sprint 4 | Python 3.12 validated |
| 2 | Thu | Sprint 5 | CI/CD updated |
| 2 | Fri | Buffer | Final testing, documentation |

### 8.2 Optimistic Timeline (5-7 days)

| Days | Sprints | Milestones |
|------|---------|------------|
| 1-2 | Sprint 1 | Meson setup |
| 3 | Sprint 2 | Core extensions |
| 4 | Sprint 3 | Sparse extensions |
| 5 | Sprint 4 | Python 3.12 |
| 6-7 | Sprint 5 | CI/CD, docs |

### 8.3 Key Milestones

- **M1**: Meson builds hello world (Sprint 1)
- **M2**: Core extensions compile (Sprint 2)
- **M3**: All extensions compile (Sprint 3)
- **M4**: All tests pass on 3.12 (Sprint 4)
- **M5**: CI/CD passing (Sprint 5)
- **M6**: Phase 2 complete, ready to merge

---

## 9. Deliverables Checklist

### 9.1 Code Deliverables
- [ ] `pyproject.toml` (PEP 517/518 compliant)
- [ ] Root `meson.build`
- [ ] `redblackgraph/meson.build`
- [ ] `redblackgraph/core/meson.build`
- [ ] `redblackgraph/sparse/meson.build`
- [ ] `redblackgraph/sparse/csgraph/meson.build`
- [ ] Converted template files
- [ ] Updated CI/CD configuration

### 9.2 Documentation Deliverables
- [ ] `BUILDING.md` (comprehensive build guide)
- [ ] Updated `README.md`
- [ ] Migration guide for developers
- [ ] Phase 2 completion summary
- [ ] Sprint completion summaries (5 sprints)
- [ ] Test results documentation

### 9.3 Testing Deliverables
- [ ] Test results: Python 3.10 (117/117)
- [ ] Test results: Python 3.11 (117/117)
- [ ] Test results: Python 3.12 (117/117)
- [ ] Performance comparison report
- [ ] Build time measurements

---

## 10. Acceptance Criteria

Phase 2 will be accepted when ALL criteria are met:

### 10.1 Functional Requirements
- ✅ All extensions build successfully with Meson
- ✅ No `numpy.distutils` imports in codebase
- ✅ All 117 tests pass on Python 3.10
- ✅ All 117 tests pass on Python 3.11
- ✅ All 117 tests pass on Python 3.12
- ✅ No regressions in functionality

### 10.2 Performance Requirements
- ✅ Build time ≤ baseline (or faster)
- ✅ Import time unchanged
- ✅ Runtime performance unchanged

### 10.3 Quality Requirements
- ✅ Code compiles without warnings
- ✅ All type checks pass
- ✅ No memory leaks (valgrind if needed)
- ✅ CI/CD passing on all platforms

### 10.4 Documentation Requirements
- ✅ Build process documented
- ✅ Developer setup guide updated
- ✅ Migration guide complete
- ✅ All TODOs resolved

---

## 11. Post-Phase 2 Tasks

### 11.1 Immediate (Before Merge)
- Tag release: `v0.5.0-phase2` or `v1.0.0`
- Create GitHub release notes
- Announce Python 3.12 support

### 11.2 Short-Term (1-2 weeks)
- Monitor for issues
- Update package on PyPI
- Update documentation site (if applicable)
- Remove old setup.py files (deprecation period)

### 11.3 Future Phases
- **Phase 3**: NumPy 2.0 migration (if needed)
- **Phase 4**: Performance optimization
- **Phase 5**: Additional platform support

---

## 12. References & Resources

### 12.1 Official Documentation
- **Meson**: https://mesonbuild.com/
- **meson-python**: https://meson-python.readthedocs.io/
- **NumPy Meson Guide**: https://numpy.org/devdocs/building/
- **PEP 517**: https://peps.python.org/pep-0517/
- **PEP 518**: https://peps.python.org/pep-0518/

### 12.2 Example Projects
- **NumPy**: https://github.com/numpy/numpy
- **SciPy**: https://github.com/scipy/scipy
- **scikit-image**: https://github.com/scikit-image/scikit-image
- **pandas**: https://github.com/pandas-dev/pandas (future)

### 12.3 Community Resources
- **Scientific Python Discourse**: https://discuss.scientific-python.org/
- **Meson Discord**: https://discord.gg/meson
- **NumPy Mailing List**: numpy-discussion@python.org

---

## 13. Approval & Sign-off

| Role | Name | Status | Date |
|------|------|--------|------|
| **Specification Author** | AI Assistant | Draft | 2025-10-21 |
| **Technical Reviewer** | TBD | Pending | - |
| **Project Lead** | TBD | Pending | - |

---

## 14. Appendices

### Appendix A: Template Conversion Example

**Before (NumPy .src format)**:
```c
/**begin repeat
 * #name = byte, ubyte, short#
 * #type = npy_byte, npy_ubyte, npy_short#
 */
static @type@ @name@_function(@type@ x) {
    return x * 2;
}
/**end repeat**/
```

**After (Meson format)**:
```python
# In meson.build
types = ['byte', 'ubyte', 'short']
npy_types = ['npy_byte', 'npy_ubyte', 'npy_short']

foreach i : range(types.length())
  conf_data = configuration_data()
  conf_data.set('name', types[i])
  conf_data.set('type', npy_types[i])
  
  configure_file(
    input: 'template.c.in',
    output: '@0@_functions.c'.format(types[i]),
    configuration: conf_data
  )
endforeach
```

### Appendix B: Quick Start Commands

```bash
# Install tools
pip install meson meson-python ninja

# Setup build
meson setup builddir

# Build
meson compile -C builddir

# Install (development)
pip install -e . --no-build-isolation

# Test
pytest tests/

# Clean
rm -rf builddir build/
```

---

**Document Status**: Draft for Review  
**Version**: 1.0  
**Last Updated**: 2025-10-21  
**Next Review**: Before Sprint 1 start
