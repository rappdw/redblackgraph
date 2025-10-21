# Phase 1 Risk Assessment and Mitigation Plan

**Date**: 2025-10-21  
**Purpose**: Identify, assess, and plan mitigation for Phase 1 migration risks  
**Sprint**: Phase 1, Sprint 1

---

## Executive Summary

Phase 1 migration has **moderate overall risk** with several high-impact technical risks that have been identified and planned for. This is a research project with no production users, which significantly reduces risk severity. The primary risks involve NumPy C API compatibility and build system deprecation.

### Risk Overview

- **Total Risks Identified**: 12
- **High Severity**: 4
- **Medium Severity**: 6
- **Low Severity**: 2

### Risk Tolerance

**Project Type**: Research project  
**Risk Tolerance**: Medium-High  
**Acceptable Downtime**: N/A (no production system)  
**Rollback Capability**: Yes (via git branch management)

---

## Risk Summary Table

| # | Risk | Severity | Likelihood | Impact | Mitigation Strategy |
|---|------|----------|-----------|--------|---------------------|
| 1 | NumPy C API incompatibility | 🔴 High | Medium | Build failure | Audit C code early; test with NumPy 1.26 |
| 2 | NPY_NO_EXPORT macro missing | 🔴 High | High | Compilation error | Replace with static or remove |
| 3 | einsum internal API changes | 🔴 High | Medium | Runtime failure | Test extensively; use public API if needed |
| 4 | numpy.distutils removal | 🔴 High | Low* | Build failure | Use setuptools<60.0 temporarily |
| 5 | Cython 3.x compatibility | 🟡 Medium | Low | Build failure | Test early in Sprint 2 |
| 6 | Platform build failures | 🟡 Medium | Medium | Reduced platform support | Test on Linux and macOS |
| 7 | Test failures after migration | 🟡 Medium | Medium | Incomplete migration | Comprehensive Sprint 4 testing |
| 8 | Dependency version conflicts | 🟡 Medium | Low | Build/runtime issues | Conservative version constraints |
| 9 | Performance regression | 🟡 Medium | Low | Slower execution | Deferred to future sprints |
| 10 | Versioneer incompatibility | 🟢 Low | Low | Version string issues | Update if needed |
| 11 | Docker image issues | 🟢 Low | Low | Dev environment issues | Update in Sprint 4 |
| 12 | Documentation drift | 🟡 Medium | Medium | User confusion | Update all docs in Sprint 4 |

*Low in Phase 1 because we keep using numpy.distutils with older setuptools

---

## Technical Risks

### Risk 1: NumPy C API Incompatibility

**Severity**: 🔴 High  
**Likelihood**: Medium  
**Impact**: Build failure, non-functional C extensions

#### Description

C extension modules use NumPy C API directly. NumPy 1.26 may have changed or removed API functions used by the codebase. This could prevent compilation or cause runtime crashes.

#### Indicators

- Compilation errors referencing NumPy C API
- Undefined symbols during linking
- Segfaults when importing C extensions
- Test failures in C extension functionality

#### Detection Method

```bash
# Attempt build with NumPy 1.26
uv pip install --python .venv-3.10 "numpy>=1.26.0,<1.27"
uv pip install --python .venv-3.10 -e .

# Look for compilation errors
# Check import:
.venv-3.10/bin/python -c "import redblackgraph.core"
```

#### Mitigation Strategy

**Sprint 1**: Document all NumPy C API usage patterns  
**Sprint 2**: Test compilation with NumPy 1.26 immediately  
**Sprint 3**:
- Review NumPy 1.26 C API changelog
- Update deprecated API calls
- Add compatibility shims if needed
- Extensive testing

#### Contingency Plan

If NumPy 1.26 C API is incompatible:
1. **Option A**: Use NumPy 1.25.x temporarily (if compatible with Python 3.10+)
2. **Option B**: Rewrite C extensions to use only stable API
3. **Option C**: Switch to pure Cython implementations (already exist for some functions)

#### Status

- ✅ API patterns documented in code inventory
- 🔲 NumPy 1.26 changelog review (Sprint 2)
- 🔲 Compilation testing (Sprint 2)
- 🔲 Updates implemented (Sprint 3)

---

### Risk 2: NPY_NO_EXPORT Macro Missing

**Severity**: 🔴 High  
**Likelihood**: High  
**Impact**: All C extensions fail to compile

#### Description

C code uses `NPY_NO_EXPORT` macro which may not exist in NumPy 1.26+. This macro was used to mark functions as internal-only.

**Already Observed**: Sprint 1 testing showed this error with modern NumPy.

#### Indicators

```
error: unknown type name 'NPY_NO_EXPORT'; did you mean 'NPY_NO_SMP'?
```

#### Detection Method

Already detected during Sprint 1 environment setup.

#### Mitigation Strategy

**Sprint 3**:
1. Replace all `NPY_NO_EXPORT` with `static` keyword
2. Or remove if not needed
3. Test compilation
4. Verify functionality unchanged

**Code Change Example**:
```c
// Before:
NPY_NO_EXPORT int internal_function(...) { ... }

// After:
static int internal_function(...) { ... }
```

#### Contingency Plan

If `static` doesn't work:
- Remove export control entirely
- Use compiler-specific visibility attributes
- Use `__attribute__((visibility("hidden")))` on GCC

#### Status

- ✅ Issue identified
- ✅ Solution planned
- 🔲 Implementation (Sprint 3)
- 🔲 Testing (Sprint 3)

---

### Risk 3: einsum Internal API Changes

**Severity**: 🔴 High  
**Likelihood**: Medium  
**Impact**: Relational composition operations fail

#### Description

Code in `relational_composition.c.src` uses NumPy einsum internal APIs and debugging macros. These internals are not guaranteed stable across NumPy versions.

**Architectural Note**: "Yes, we use einsum over internal numpy matrix representation"

#### Indicators

- Compilation warnings about undeclared einsum functions
- Runtime errors in relational composition operations
- Test failures in relational_composition tests
- Segfaults during einsum operations

#### Detection Method

```bash
# Run specific tests
.venv-3.10/bin/python -m pytest tests/ -k "relational" -v

# Check for warnings:
.venv-3.10/bin/python -W all -c "import redblackgraph.core; # test einsum"
```

#### Mitigation Strategy

**Sprint 3**:
1. Identify exactly which einsum internals are used
2. Check NumPy 1.26 einsum implementation
3. Test all relational composition operations
4. If internals changed:
   - Option A: Update to use new internal API
   - Option B: Switch to public `np.einsum()` function
   - Option C: Rewrite using alternative approach

#### Contingency Plan

1. **Fallback to Cython**: `_relational_composition.pyx` exists - switch to that
2. **Rewrite**: Implement using standard NumPy operations
3. **Pin NumPy**: If necessary, pin to last working NumPy version

#### Status

- ✅ Risk identified
- 🔲 einsum usage audit (Sprint 3)
- 🔲 NumPy 1.26 testing (Sprint 3)
- 🔲 Mitigation if needed (Sprint 3)

---

### Risk 4: numpy.distutils Removal

**Severity**: 🔴 High  
**Likelihood**: Low (in Phase 1)  
**Impact**: Cannot build at all

#### Description

numpy.distutils was deprecated in NumPy 1.23.0 and will be removed in NumPy 2.0. It also requires Python distutils which was removed in Python 3.12.

**Phase 1 Status**: Keeping numpy.distutils by using:
- NumPy < 2.0 (1.26.x)
- setuptools < 60.0 (still includes distutils shim)

**Phase 2 Status**: Must migrate to Meson

#### Indicators

```
DeprecationWarning: `numpy.distutils` is deprecated since NumPy 1.23.0
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

#### Detection Method

Already detected - this is a known issue.

#### Mitigation Strategy

**Phase 1 (Sprints 2-4)**:
1. Pin setuptools<60.0 to keep distutils available
2. Pin numpy<2.0 to keep numpy.distutils available
3. Accept deprecation warnings
4. Plan for Phase 2 Meson migration

**Phase 2** (future):
1. Replace numpy.distutils with Meson build system
2. Write meson.build files
3. Handle .c.src template processing manually or with Meson
4. Full migration away from numpy.distutils

#### Contingency Plan

If setuptools<60.0 doesn't work:
1. Install distutils separately: `pip install setuptools distutils`
2. Use older NumPy temporarily
3. Accelerate Phase 2 Meson migration

#### Status

- ✅ Issue understood
- ✅ Phase 1 workaround identified (setuptools<60.0)
- 🔲 Implementation (Sprint 2)
- 🔲 Phase 2 planning (post-Phase 1)

---

### Risk 5: Cython 3.x Compatibility

**Severity**: 🟡 Medium  
**Likelihood**: Low  
**Impact**: Cython extensions fail to compile

#### Description

Project has 7 Cython extensions. Cython 3.0 introduced breaking changes. Need to verify compatibility.

**Current**: No Cython version constraint  
**Target**: Cython >= 3.0

#### Indicators

- Cython compilation errors
- Deprecation warnings during Cython build
- Generated C code doesn't compile
- Import errors on Cython modules

#### Detection Method

```bash
# Test Cython compilation
uv pip install --python .venv-3.10 "cython>=3.0"
# Attempt build (in Sprint 2)
```

#### Mitigation Strategy

**Sprint 2**: Test compilation immediately after dependency updates  
**Sprint 3**: Fix any Cython 3.x incompatibilities  
- Review Cython 3.0 migration guide
- Update Cython syntax if needed
- Test all 7 .pyx files

#### Contingency Plan

1. Pin to older Cython if blocking (Cython 0.29.x)
2. Update Cython syntax manually
3. Switch to pure C if necessary (unlikely)

#### Status

- ✅ Risk identified
- 🔲 Cython 3.x testing (Sprint 2)
- 🔲 Fixes if needed (Sprint 3)

---

### Risk 6: Platform Build Failures

**Severity**: 🟡 Medium  
**Likelihood**: Medium  
**Impact**: Reduced platform support

#### Description

Code must build on both Linux and macOS. Platform-specific issues could arise from:
- Different compilers (gcc vs clang)
- Different system libraries
- Architecture differences

**Dev Environment**: Linux aarch64  
**CI/CD**: Travis CI (Linux x86_64)  
**Target**: Also needs macOS

#### Indicators

- Build succeeds on Linux, fails on macOS (or vice versa)
- Different test failures on different platforms
- Platform-specific compiler errors

#### Detection Method

- Travis CI will test Linux x86_64
- Manual testing needed for macOS (if available)
- Dev environment tests Linux aarch64

#### Mitigation Strategy

**Sprint 2**: Test builds on all available platforms early  
**Sprint 4**: Comprehensive platform testing  
**Continuous**: Monitor Travis CI results

#### Contingency Plan

If platform-specific issues:
1. Add platform-specific code with conditionals
2. Adjust compiler flags per platform
3. Document platform limitations if unfixable

#### Status

- ✅ Risk identified
- 🔲 Multi-platform testing (Sprint 2+)

---

### Risk 7: Test Failures After Migration

**Severity**: 🟡 Medium  
**Likelihood**: Medium  
**Impact**: Incomplete migration, potential bugs

#### Description

Changes to dependencies and code could introduce bugs or break existing functionality. Tests may fail due to:
- NumPy behavioral changes
- SciPy behavioral changes
- Subtle compatibility issues

#### Indicators

- Tests that pass on Python 3.8 fail on Python 3.10+
- Numerical results differ slightly
- Edge cases break

#### Detection Method

**Sprint 4**: Comprehensive test execution
```bash
for v in 3.10 3.11 3.12; do
  .venv-${v}/bin/python -m pytest tests/ -v
done
```

#### Mitigation Strategy

**Sprint 1**: Establish baseline (blocked due to build issues)  
**Sprint 4**: 
- Run full test suite on all Python versions
- Investigate and fix all failures
- Add new tests if gaps found
- Compare behavior to baseline where possible

#### Contingency Plan

If tests fail and can't be fixed quickly:
1. Investigate if test expectations need updating (not a bug)
2. Document known issues
3. Defer non-critical fixes to post-Phase 1
4. Roll back if failures are critical

#### Status

- ✅ Risk identified
- 🔲 Testing (Sprint 4)

---

### Risk 8: Dependency Version Conflicts

**Severity**: 🟡 Medium  
**Likelihood**: Low  
**Impact**: Build or runtime issues

#### Description

Dependency version constraints may conflict:
- NumPy 1.26 vs SciPy 1.11 compatibility
- Cython 3.x vs NumPy 1.26 compatibility
- fs-crawler compatibility with Python 3.10+

#### Indicators

- Pip resolver cannot find compatible versions
- Runtime import errors
- Version conflicts during install

#### Detection Method

```bash
uv pip install --python .venv-3.10 -e ".[dev,test]"
# Check for resolution errors
```

#### Mitigation Strategy

**Sprint 2**: Use conservative version constraints  
- NumPy: `>=1.26.0,<1.27` (not 2.0)
- SciPy: `>=1.11.0` (no upper bound initially)
- Test all combinations

**Verification**: Per architectural clarification, fs-crawler>=0.3.2 works with Python 3.10+

#### Contingency Plan

1. Adjust version ranges if conflicts
2. Test with wider range if needed
3. Pin to specific versions temporarily

#### Status

- ✅ Risk identified
- ✅ Conservative approach planned
- 🔲 Implementation (Sprint 2)

---

### Risk 9: Performance Regression

**Severity**: 🟡 Medium  
**Likelihood**: Low  
**Impact**: Slower execution

#### Description

NumPy 1.26 or Python 3.10+ may have different performance characteristics. Operations could be slower.

**Sprint 1 Decision**: Performance benchmarking deferred

#### Indicators

- Operations take noticeably longer
- Memory usage increases
- Algorithmic complexity changes

#### Detection Method

Not being measured in Phase 1 (per architectural decision).

#### Mitigation Strategy

**Phase 1**: Accept performance as-is  
**Post-Phase 1**: If users report issues, investigate and optimize

#### Contingency Plan

If performance regression discovered later:
1. Profile to find bottleneck
2. Optimize hot paths
3. Consider algorithmic improvements
4. Document performance characteristics

#### Status

- ✅ Risk accepted
- 🔲 Deferred to post-Phase 1

---

### Risk 10: Versioneer Incompatibility

**Severity**: 🟢 Low  
**Likelihood**: Low  
**Impact**: Version string generation fails

#### Description

Versioneer might not work correctly with updated build system.

#### Indicators

- Version shows as "0+unknown"
- Build fails due to versioneer error
- Git tags not recognized

#### Detection Method

```bash
.venv-3.10/bin/python -c "import redblackgraph; print(redblackgraph.__version__)"
```

#### Mitigation Strategy

**Sprint 2**: Check versioneer version and update if available  
**Testing**: Verify version generation works after build

#### Contingency Plan

1. Update to latest versioneer
2. Manually set version if needed
3. Switch to simpler versioning (e.g., `__version__ = "2.0.0"`)

#### Status

- ✅ Risk identified
- 🔲 Versioneer check (Sprint 2)

---

### Risk 11: Docker Image Issues

**Severity**: 🟢 Low  
**Likelihood**: Low  
**Impact**: Dev environment issues (non-blocking)

#### Description

Docker images may need updates for Python 3.10+.

#### Indicators

- Docker build fails
- Tests fail inside Docker
- Image size increases significantly

#### Detection Method

```bash
# In Sprint 4
docker build -t redblackgraph:test .
docker run redblackgraph:test python -m pytest tests/
```

#### Mitigation Strategy

**Sprint 4**: Update Dockerfiles to use Python 3.10+  
**Low Priority**: Not blocking for core functionality

#### Contingency Plan

Docker issues can be deferred to post-Phase 1 if time-constrained.

#### Status

- ✅ Risk identified
- 🔲 Docker updates (Sprint 4)

---

### Risk 12: Documentation Drift

**Severity**: 🟡 Medium  
**Likelihood**: Medium  
**Impact**: User confusion

#### Description

Documentation may not reflect new Python versions, installation steps, or API changes.

#### Indicators

- README shows old Python versions
- Installation instructions don't work
- Examples use deprecated APIs

#### Detection Method

Manual review of all documentation files.

#### Mitigation Strategy

**Sprint 4**: 
- Update README.md
- Update Jupyter notebooks
- Verify all examples work
- Update installation instructions

#### Contingency Plan

Documentation updates can be completed post-Phase 1 if necessary.

#### Status

- ✅ Risk identified
- 🔲 Documentation updates (Sprint 4)

---

## Rollback Procedures

### Sprint-Level Rollback

If a sprint fails critically:

```bash
# Find the previous sprint's completion tag
git tag -l

# Reset to that tag
git reset --hard phase1-sprint1-complete

# Force push (since we're on migration branch)
git push origin migration --force
```

### Phase-Level Rollback

If Phase 1 must be abandoned:

```bash
# Backup current work
git branch migration-backup-$(date +%Y%m%d) migration

# Delete migration branch
git checkout master
git branch -D migration

# Old versions remain on PyPI for users
# No impact to existing users
```

### Criteria for Rollback

**Sprint Rollback**:
- Cannot resolve critical blocker within 2 days
- Technical approach proven infeasible
- Discovered fundamental incompatibility

**Phase Rollback**:
- Multiple sprints fail
- NumPy 1.26 fundamentally incompatible
- Cannot maintain required functionality
- Cost/benefit ratio unfavorable

---

## Communication Plan

### Issue Escalation

**Level 1 - Minor Issue** (< 4 hours to resolve):
- Document in sprint notes
- Resolve independently
- Mention in next status update

**Level 2 - Moderate Issue** (4-24 hours to resolve):
- Document in `.plans/sprint_X_issues.md`
- Notify project owner
- Adjust sprint timeline if needed

**Level 3 - Major Issue** (> 1 day to resolve):
- Stop work on affected tasks
- Document blocker comprehensively
- Contact project owner immediately
- Discuss rollback vs push forward

**Level 4 - Critical Issue** (phase-blocking):
- Halt all Phase 1 work
- Emergency consultation with project owner
- Evaluate rollback necessity
- Document lessons learned

### Project Owner Contact

**Name**: Daniel Rapp  
**Email**: rappdw@gmail.com  
**Role**: Final decision authority

### Status Updates

**Frequency**: As needed (research project, flexible)  
**Format**: Live demo at sprint completion  
**Channel**: Direct communication

---

## Success Criteria

### Sprint Completion Criteria

**Sprint 1**: ✅ Foundation established, risks documented  
**Sprint 2**: Dependencies updated, builds work  
**Sprint 3**: Code modernized, extensions compile  
**Sprint 4**: All tests pass, ready for merge

### Phase 1 Go/No-Go Decision Criteria

**GO** if:
- ✅ All tests pass on Python 3.10, 3.11, 3.12
- ✅ Builds succeed on target platforms
- ✅ No critical functionality lost
- ✅ No critical bugs introduced

**NO-GO** if:
- ❌ Tests fail and cannot be fixed
- ❌ Critical functionality broken
- ❌ Performance regression > 50%
- ❌ Cannot build reliably

---

## Lessons Learned (To Be Updated)

### Sprint 1 Lessons

- numpy.distutils deprecation more impactful than initially thought
- Modern NumPy incompatibility discovered early (good)
- uv works well for Python version management
- C API issues identifiable through early build attempts

### Future Lessons

*To be filled in after each sprint*

---

## Appendices

### A. Related Documents

- `.plans/phase1_baseline_report.md` - Current state assessment
- `.plans/phase1_code_inventory.md` - Code requiring changes
- `.plans/phase1_modification_checklist.md` - Detailed task list

### B. External References

- [NumPy 1.26 Release Notes](https://numpy.org/devdocs/release/1.26.0-notes.html)
- [Cython 3.0 Migration Guide](https://cython.readthedocs.io/en/latest/src/userguide/migrating_to_cy30.html)
- [numpy.distutils Status](https://numpy.org/devdocs/reference/distutils_status_migration.html)

---

**Document Status**: Complete  
**Last Updated**: 2025-10-21  
**Owner**: Tech Lead / PM  
**Next Review**: After each sprint completion
