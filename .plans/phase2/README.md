# Phase 2: Meson Build System Migration

**Branch**: `phase2-meson`  
**Based On**: `master` (v0.4.0-phase1)  
**Goal**: Python 3.12+ support via Meson build system  
**Status**: 🚧 Planning

---

## Objective

Migrate RedBlackGraph from `numpy.distutils` (deprecated) to Meson build system to enable Python 3.12+ support.

---

## Background

**Why Meson?**
- Python 3.12 removed `distutils` from stdlib (PEP 632)
- `numpy.distutils` deprecated in NumPy 1.26, removed in NumPy 2.0
- Meson is the official NumPy/SciPy replacement
- Used by NumPy, SciPy, scikit-image, and other major projects

**Current Blocker:**
RedBlackGraph uses `numpy.distutils` in 5 files:
- `/setup.py`
- `/redblackgraph/setup.py`
- `/redblackgraph/core/setup.py`
- `/redblackgraph/sparse/setup.py`
- `/redblackgraph/sparse/csgraph/setup.py`

---

## Target Versions

| Component | Version |
|-----------|---------|
| Python | 3.10, 3.11, **3.12** |
| NumPy | 1.26+ |
| SciPy | 1.15+ |
| Meson | Latest |
| meson-python | Latest |

---

## Sprint Plan

### Sprint 1: Meson Setup & Learning (2-3 days)
- [ ] Install Meson toolchain
- [ ] Study NumPy/SciPy meson.build examples
- [ ] Create minimal root `meson.build`
- [ ] Create `pyproject.toml`
- [ ] Test basic build

### Sprint 2: Core Extension Migration (1-2 days)
- [ ] Create `redblackgraph/core/meson.build`
- [ ] Convert `.src` file templating to Meson
- [ ] Configure NumPy C API includes
- [ ] Build and test core extensions

### Sprint 3: Sparse Extension Migration (1-2 days)
- [ ] Create `redblackgraph/sparse/meson.build`
- [ ] Migrate Cython modules
- [ ] Configure C++ sparsetools
- [ ] Implement symbol visibility

### Sprint 4: Python 3.12 Testing (1 day)
- [ ] Create Python 3.12 venv
- [ ] Build with meson-python
- [ ] Run full test suite (target: 117/117 passing)
- [ ] Fix any Python 3.12-specific issues

### Sprint 5: CI/CD & Documentation (1 day)
- [ ] Update CI/CD for Meson
- [ ] Update build documentation
- [ ] Create BUILDING.md
- [ ] Tag Phase 2 completion

---

## Estimated Timeline

- **Conservative**: 11 days (~2.5 weeks)
- **Optimistic**: 5-7 days (~1.5 weeks)

---

## Success Criteria

Phase 2 is complete when:
- ✅ All C/C++ extensions build with Meson
- ✅ All Cython modules compile
- ✅ Python 3.10, 3.11, 3.12 all supported
- ✅ All 117 tests pass on all versions
- ✅ No numpy.distutils dependencies remain
- ✅ CI/CD working
- ✅ Documentation updated

---

## Resources

- **Detailed Roadmap**: `..//python312_migration_analysis.md`
- **NumPy Meson Guide**: https://numpy.org/devdocs/building/bldstr_meson.html
- **SciPy Examples**: https://github.com/scipy/scipy/tree/main/scipy
- **Meson Docs**: https://mesonbuild.com/

---

## Current Status

**Phase**: Planning  
**Next Action**: Start Sprint 1 - Meson Setup & Learning

---

**Created**: 2025-10-21  
**Last Updated**: 2025-10-21
