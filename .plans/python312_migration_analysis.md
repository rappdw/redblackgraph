# Python 3.12 Support - Migration Analysis & Roadmap

**Date**: 2025-10-21  
**Current Status**: Python 3.10 & 3.11 ✅ | Python 3.12 ⚠️ Blocked  
**Target**: Python 3.12 Support

---

## Executive Summary

Python 3.12 support is **currently blocked** due to the removal of `distutils` from the Python standard library (PEP 632). RedBlackGraph uses `numpy.distutils` extensively for building C/C++ extensions, which itself depends on `distutils`.

**Key Fact**: `numpy.distutils` was **deprecated in NumPy 1.26** and **removed in NumPy 2.0**, affecting Python 3.12+ compatibility.

---

## Current Build System Analysis

### Dependencies on numpy.distutils

**Files using numpy.distutils:**
1. `/setup.py` (lines 12, 36, 158)
2. `/redblackgraph/setup.py` (lines 5, 17)
3. `/redblackgraph/core/setup.py` (lines 7, 27)
4. `/redblackgraph/sparse/setup.py` (lines 9, 42)
5. `/redblackgraph/sparse/csgraph/setup.py` (line 3)

**What numpy.distutils provides:**
- `Configuration` class for organizing multi-package builds
- Custom `build_ext` command for C/C++ extension compilation
- Fortran support (not used in this project)
- Template file processing (`.src` files for NumPy-style templating)
- Integration with NumPy's C API

### Custom Build Features Used

1. **`.src` file templating** - NumPy's templating system for generating repetitive C code
   - `redblackgraph/core/src/redblackgraph/*.c.src`
   - `redblackgraph/core/src/redblackgraph/*.h.src`

2. **Custom build_ext** - Version script for symbol hiding on GNU linkers
   - Prevents symbol conflicts in shared libraries
   - Linux-specific optimization

3. **Multi-package configuration** - Hierarchical setup.py structure
   - Main `setup.py` → `redblackgraph/setup.py` → `core/setup.py`, `sparse/setup.py`

4. **Cython integration** - Pre-build Cython code generation
   - 7 Cython modules in `sparse/` and `core/`

---

## Migration Options

### Option 1: Meson Build System (RECOMMENDED)

**Status**: ✅ **Industry Standard for NumPy/SciPy Projects**

#### Description
Meson is the official replacement for `numpy.distutils` and is used by NumPy, SciPy, and other scientific Python projects.

#### Pros
- ✅ **Official NumPy recommendation**
- ✅ **Actively maintained** by NumPy/SciPy teams
- ✅ **Faster builds** than distutils
- ✅ **Better dependency tracking**
- ✅ **Cross-platform** support
- ✅ **Modern build system** (not deprecated)
- ✅ **Good documentation** and examples from NumPy/SciPy

#### Cons
- ⚠️ **Learning curve** - different syntax than setup.py
- ⚠️ **Migration effort** - requires rewriting build configuration
- ⚠️ **New dependency** - requires Meson and ninja-build

#### Effort Estimate
**Time**: 3-5 days  
**Complexity**: Medium-High  
**Risk**: Low (well-tested by major projects)

#### Files to Create/Modify
- Create `meson.build` (root)
- Create `redblackgraph/meson.build`
- Create `redblackgraph/core/meson.build`
- Create `redblackgraph/sparse/meson.build`
- Create `pyproject.toml` (PEP 517/518 build config)
- Migrate `.src` templating to Meson's templating
- Modify CI/CD scripts

---

### Option 2: scikit-build-core (ALTERNATIVE)

**Status**: ✅ **CMake-based Build System**

#### Description
scikit-build-core provides a bridge between Python packaging and CMake, supporting modern Python build standards.

#### Pros
- ✅ **Modern PEP 517/518 support**
- ✅ **CMake ecosystem** (widely used)
- ✅ **Good for complex C/C++ projects**
- ✅ **Active development**

#### Cons
- ⚠️ **CMake learning curve**
- ⚠️ **More complex** than Meson for simple cases
- ⚠️ **Not the NumPy standard** (less Python-scientific community adoption)
- ⚠️ **Requires CMake knowledge**

#### Effort Estimate
**Time**: 4-6 days  
**Complexity**: High  
**Risk**: Medium

---

### Option 3: Pure setuptools + Cython (SIMPLER BUT LIMITED)

**Status**: ⚠️ **Possible but Loses Features**

#### Description
Use vanilla setuptools with Cython's build_ext, avoiding numpy.distutils entirely.

#### Pros
- ✅ **Simpler migration**
- ✅ **No new build tool**
- ✅ **Works with Python 3.12**

#### Cons
- ❌ **Lose `.src` templating** - must manually expand templates
- ❌ **Lose custom build_ext features** - symbol hiding
- ❌ **Less NumPy C API integration**
- ❌ **Manual work** to expand all `.src` files

#### Effort Estimate
**Time**: 2-3 days  
**Complexity**: Low-Medium  
**Risk**: High (loss of functionality)

---

### Option 4: Keep numpy.distutils (Pin NumPy <2.0) - NOT RECOMMENDED

**Status**: ❌ **Temporary Workaround Only**

#### Description
Pin NumPy to <2.0 to keep using numpy.distutils.

#### Pros
- ✅ **No migration needed**
- ✅ **Works immediately**

#### Cons
- ❌ **Blocks Python 3.12 support** (NumPy 2.0 is designed for Py 3.12+)
- ❌ **Deprecated and removed** technology
- ❌ **No future support**
- ❌ **Incompatible with modern ecosystem**
- ❌ **Technical debt**

#### Verdict
**DO NOT USE** - Only for emergency backward compatibility.

---

## Recommended Approach: Meson Migration

### Phase 2 Roadmap (Python 3.12 Support)

#### Sprint 1: Meson Setup & Learning (2-3 days)

**Goals:**
1. Install Meson and ninja-build
2. Study NumPy/SciPy meson.build files
3. Create minimal meson.build for RedBlackGraph
4. Test basic build on Python 3.11 (verify it works)

**Tasks:**
- [ ] Install Meson (`pip install meson ninja meson-python`)
- [ ] Review NumPy meson migration guide
- [ ] Review SciPy meson.build structure
- [ ] Create root `meson.build`
- [ ] Create `pyproject.toml`
- [ ] Build minimal extension (test)

**References:**
- NumPy Meson Guide: https://numpy.org/devdocs/building/bldstr_meson.html
- SciPy Meson Examples: https://github.com/scipy/scipy/tree/main/scipy
- Meson Python Integration: https://mesonbuild.com/Python-module.html

---

#### Sprint 2: Core Extension Migration (1-2 days)

**Goals:**
1. Migrate `redblackgraph/core` C extensions to Meson
2. Handle `.src` file templating
3. Test core extension builds

**Tasks:**
- [ ] Create `redblackgraph/core/meson.build`
- [ ] Convert `.src` templating to Meson equivalents
- [ ] Configure NumPy C API includes
- [ ] Build and test core extensions
- [ ] Verify Python import works

**Template Conversion:**
```python
# NumPy .src files use @name@ substitution
# Meson uses configuration_data() and configure_file()

# Example:
conf_data = configuration_data()
conf_data.set('name', 'byte')
conf_data.set('type', 'npy_byte')
configure_file(
    input: 'rbg_math.c.src',
    output: 'rbg_math_byte.c',
    configuration: conf_data
)
```

---

#### Sprint 3: Sparse Extension Migration (1-2 days)

**Goals:**
1. Migrate `redblackgraph/sparse` C/C++ extensions
2. Handle Cython modules
3. Configure symbol visibility

**Tasks:**
- [ ] Create `redblackgraph/sparse/meson.build`
- [ ] Migrate Cython modules to Meson
- [ ] Configure C++ sparsetools compilation
- [ ] Implement version script for symbol hiding
- [ ] Build and test sparse extensions

**Cython in Meson:**
```python
# meson.build
cython = find_program('cython')
rbm_pyx = custom_target(
    'rbm_c',
    output: '_rbm.c',
    input: 'rbm.pyx',
    command: [cython, '-3', '@INPUT@', '-o', '@OUTPUT@']
)
```

---

#### Sprint 4: Python 3.12 Testing (1 day)

**Goals:**
1. Create Python 3.12 venv
2. Build with Meson
3. Run full test suite
4. Fix any Python 3.12-specific issues

**Tasks:**
- [ ] Create .venv-3.12
- [ ] Install dependencies
- [ ] Build with meson-python
- [ ] Run test suite
- [ ] Verify all 117 tests pass
- [ ] Check for Python 3.12 deprecations

---

#### Sprint 5: CI/CD & Documentation (1 day)

**Goals:**
1. Update CI/CD for Meson builds
2. Document build process
3. Update development setup guide

**Tasks:**
- [ ] Update `.travis.yml` or migrate to GitHub Actions
- [ ] Update `README.md` build instructions
- [ ] Create BUILDING.md with Meson details
- [ ] Update developer documentation
- [ ] Tag Phase 2 completion

---

## Immediate Next Steps (If Starting Today)

### 1. Research & Planning (1-2 hours)

```bash
# Study existing Meson projects
git clone https://github.com/numpy/numpy.git numpy-reference
git clone https://github.com/scipy/scipy.git scipy-reference

# Look at their meson.build files
cat numpy-reference/meson.build
cat scipy-reference/scipy/meson.build
```

### 2. Create Minimal pyproject.toml (30 minutes)

```toml
# pyproject.toml
[build-system]
requires = ["meson-python", "Cython>=3.0", "numpy>=1.26"]
build-backend = "mesonpy"

[project]
name = "redblackgraph"
version = "0.3.17"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26.0,<2.0",
    "scipy>=1.11.0",
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]
```

### 3. Create Root meson.build (1 hour)

```python
# meson.build
project('redblackgraph',
  'c', 'cpp', 'cython',
  version: '0.3.17',
  default_options: [
    'buildtype=debugoptimized',
  ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

subdir('redblackgraph')
```

### 4. Test Minimal Build (1 hour)

```bash
# Install meson-python
pip install meson-python ninja meson

# Try building (will fail but shows what's needed)
python -m build --no-isolation

# Or use meson directly
meson setup build
ninja -C build
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Breaking changes during migration** | HIGH | Extensive testing, keep Python 3.10/3.11 working |
| **Loss of build features** | MEDIUM | Study Meson docs, replicate all features |
| **CI/CD disruption** | MEDIUM | Set up parallel CI, test before switching |
| **Developer workflow changes** | LOW | Good documentation, training |
| **Third-party tool compatibility** | LOW | Meson is widely supported |

---

## Decision Matrix

| Criteria | Meson | scikit-build | Pure setuptools | Pin NumPy <2.0 |
|----------|-------|--------------|-----------------|----------------|
| **Python 3.12 Support** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Future-proof** | ✅ Best | ✅ Good | ⚠️ Basic | ❌ Deprecated |
| **NumPy/SciPy Standard** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Migration Effort** | ⚠️ Medium | ⚠️ High | ✅ Low | ✅ None |
| **Feature Preservation** | ✅ Full | ✅ Full | ❌ Partial | ✅ Full |
| **Build Performance** | ✅ Fast | ✅ Fast | ⚠️ Slower | ⚠️ Slower |
| **Community Support** | ✅ Excellent | ⚠️ Good | ⚠️ Basic | ❌ None |

**WINNER: Meson** ✅

---

## Estimated Timeline

### Conservative Estimate
- **Sprint 1**: 3 days
- **Sprint 2**: 2 days
- **Sprint 3**: 2 days
- **Sprint 4**: 1 day
- **Sprint 5**: 1 day
- **Buffer**: 2 days
- **TOTAL**: **11 days (~2.5 weeks)**

### Optimistic Estimate
- **Total**: **5-7 days (~1.5 weeks)**

---

## Success Criteria

Phase 2 is complete when:
- ✅ All C/C++ extensions build with Meson
- ✅ All Cython modules compile
- ✅ Python 3.10, 3.11, 3.12 all supported
- ✅ All 117 tests pass on all versions
- ✅ Build is faster or same speed as before
- ✅ CI/CD working
- ✅ Documentation updated
- ✅ No numpy.distutils dependencies remain

---

## Alternatives to Consider

### Hybrid Approach
Keep setup.py for Python 3.10/3.11 (backward compat), add Meson for 3.12+.

**Pros**: Gradual migration  
**Cons**: Maintain two build systems  
**Verdict**: ⚠️ Only if timeline is critical

---

## Resources & References

### Official Documentation
- **Meson**: https://mesonbuild.com/
- **Meson Python**: https://meson-python.readthedocs.io/
- **NumPy Meson Guide**: https://numpy.org/devdocs/building/
- **SciPy Meson Migration**: https://github.com/scipy/scipy/wiki/Meson

### Example Projects
- **NumPy**: https://github.com/numpy/numpy
- **SciPy**: https://github.com/scipy/scipy
- **scikit-image**: https://github.com/scikit-image/scikit-image

### Community
- **NumPy Mailing List**: numpy-discussion@python.org
- **SciPy Discourse**: https://discuss.scientific-python.org/

---

## Conclusion & Recommendation

**RECOMMENDED PATH**: Migrate to Meson in Phase 2

**Rationale**:
1. ✅ It's the **official NumPy/SciPy standard**
2. ✅ **Future-proof** - not deprecated
3. ✅ **Well-supported** by scientific Python community
4. ✅ **Faster builds** than distutils
5. ✅ **Enables Python 3.12+ support**
6. ⚠️ Medium effort (11 days) is **worth the long-term benefits**

**Alternative**: If timeline is absolutely critical, use pure setuptools with manual `.src` expansion (5 days), but this creates technical debt.

**DO NOT**: Pin NumPy <2.0 - this blocks ecosystem progress.

---

## Next Action

**Create Phase 2 Planning Document** with:
1. Detailed Meson migration tasks
2. Week-by-week timeline
3. Testing checklist
4. Rollback plan

Then **start Sprint 1: Meson Setup & Learning** when ready to proceed.

---

**Document Status**: Draft for Review  
**Last Updated**: 2025-10-21  
**Author**: Engineering Migration Team  
**Approver**: TBD
