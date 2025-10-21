# RedBlackGraph: Python, NumPy, and SciPy Upgrade Plan

## Current State

### Version Requirements
- **Python**: >= 3.6 (currently testing 3.6, 3.7, 3.8 in Travis CI)
- **NumPy**: >= 0.14.0 (extremely outdated, circa 2017)
- **SciPy**: No version constraint
- **Cython**: Used but version unspecified

### Build System
- Uses **numpy.distutils** extensively (DEPRECATED in NumPy 1.26.0, REMOVED in NumPy 2.0.0)
- Multiple C extensions with templating (`.c.src` files)
- Multiple Cython extensions (`.pyx` files in `sparse/csgraph/`)
- Custom build process with `cythonize.py` script

### Dependencies on NumPy APIs
- Uses `numpy.distutils.misc_util.Configuration`
- Uses `numpy.distutils.command.build_ext`
- Uses `numpy.distutils.core.setup`
- Uses NumPy C API for extensions
- Uses NumPy templating system (`.src` files)

### Dependencies on SciPy APIs
- `scipy.sparse.csr.csr_matrix` - base class for `rb_matrix`
- `scipy.sparse.sputils` - for `get_index_dtype`, `upcast`
- Standard sparse matrix operations

## Target State

### Recommended Versions
- **Python**: 3.9+ (drop 3.6, 3.7, 3.8 as they're EOL or near EOL)
  - Python 3.9: EOL October 2025
  - Python 3.10: EOL October 2026
  - Python 3.11: EOL October 2027
  - Python 3.12: EOL October 2028 (Recommended minimum)
- **NumPy**: 1.26.x or 2.0.x
  - NumPy 1.26.x: Last version supporting legacy features, good transition point
  - NumPy 2.0.x: Modern version with breaking changes but better performance
- **SciPy**: 1.11+ (requires NumPy 1.21.6+)
  - SciPy 1.11: Works with NumPy 1.x and 2.x
  - SciPy 1.13+: Latest with best compatibility

## Migration Strategy

### Phase 1: Build System Migration (CRITICAL)

**Problem**: `numpy.distutils` is removed in NumPy 2.0

**Solution Options**:

#### Option A: Meson + meson-python (Recommended for NumPy 2.0+)
- **Pros**: 
  - Official replacement recommended by NumPy
  - Better performance, modern build system
  - Used by NumPy, SciPy themselves
  - Better cross-platform support
- **Cons**: 
  - Requires complete rewrite of build configuration
  - Steeper learning curve
  - More complex initial setup

#### Option B: setuptools + scikit-build-core
- **Pros**: 
  - CMake-based, familiar to many developers
  - Good for mixed C/C++/Cython projects
- **Cons**: 
  - Requires CMake knowledge
  - Still significant rewrite

#### Option C: Stay on NumPy 1.26.x temporarily
- **Pros**: 
  - Minimal changes initially
  - `numpy.distutils` still available but deprecated
  - Time to plan migration
- **Cons**: 
  - Only a temporary solution
  - NumPy 1.26 is in maintenance mode
  - Eventually must migrate anyway

**Recommended Approach**: **Phased migration via Option C → Option A**
1. First upgrade to NumPy 1.26.x + SciPy 1.11+ + Python 3.10+
2. Then migrate build system to Meson
3. Finally upgrade to NumPy 2.0+

### Phase 2: Python Version Upgrade

**Current**: Python 3.6+  
**Target**: Python 3.10+ (or 3.12+ for longer support)

**Changes Required**:
1. Update `setup.py`:
   - Change `python_requires='>=3.6'` to `python_requires='>=3.10'`
   - Remove classifiers for Python 3.5, 3.6, 3.7, 3.8, 3.9
   - Add classifiers for Python 3.10, 3.11, 3.12
2. Remove `dataclasses` backport (built-in since Python 3.7)
3. Update `.travis.yml` or migrate to GitHub Actions
4. Update README badges

**Code Review Required**:
- Check for deprecated Python features
- Review async/await usage if any
- Check string formatting (f-strings are good)

### Phase 3: NumPy Version Upgrade

#### Step 3A: Upgrade to NumPy 1.26.x

**Changes Required**:
1. Update `setup.py`:
   - Change `numpy>=0.14.0` to `numpy>=1.26.0,<2.0.0`
   - Change `setup_requires` similarly
2. Update `requirements.txt`:
   - `numpy>=1.26.0,<2.0.0`
3. Address deprecation warnings:
   - Run tests and fix any deprecation warnings
   - Update C API usage if needed

**Testing Checklist**:
- [ ] All Cython extensions compile
- [ ] All C extensions compile
- [ ] All tests pass
- [ ] Matrix operations work correctly
- [ ] Sparse matrix operations work correctly

#### Step 3B: Migrate Build System to Meson

**Files to Create**:
1. `pyproject.toml` - Main build configuration
2. `meson.build` - Root build file
3. `redblackgraph/meson.build` - Package build file
4. `redblackgraph/core/meson.build` - Core extension build
5. `redblackgraph/sparse/meson.build` - Sparse extension build
6. `redblackgraph/sparse/csgraph/meson.build` - Csgraph build

**Files to Modify**:
1. Remove `numpy.distutils` imports from all `setup.py` files
2. Update Cython compilation process
3. Update C extension compilation process
4. Handle `.c.src` templating (may need custom script)

**Files to Remove** (after migration):
- `setup.py` (replaced by `pyproject.toml` + `meson.build`)
- `redblackgraph/setup.py`
- `redblackgraph/core/setup.py`
- `redblackgraph/sparse/setup.py`
- `redblackgraph/sparse/csgraph/setup.py`

#### Step 3C: Upgrade to NumPy 2.0+ (After Meson migration)

**Major Breaking Changes**:
1. **NumPy 2.0 API changes**:
   - C API changes (ABI stability improved)
   - Some functions removed/renamed
   - `numpy.ndarray` changes
   
2. **Update Code**:
   - Review all C extension code for NumPy 2.0 compatibility
   - Update Cython code for NumPy 2.0
   - Check array creation, dtype handling
   
3. **Update Requirements**:
   - `numpy>=2.0.0`

### Phase 4: SciPy Version Upgrade

**Current**: No constraint  
**Target**: scipy>=1.11.0 (or 1.13.0+ for latest)

**Changes Required**:
1. Update `setup.py` and `requirements.txt`:
   - Add `scipy>=1.11.0` or `scipy>=1.13.0`
2. Review SciPy API changes:
   - `scipy.sparse` changes (minimal expected)
   - Check `get_index_dtype`, `upcast` compatibility
   - Verify `csr_matrix` behavior

**Testing Checklist**:
- [ ] `rb_matrix` class works correctly
- [ ] Sparse matrix multiplication works
- [ ] All sparse operations function properly
- [ ] Integration tests pass

### Phase 5: CI/CD Updates

**Travis CI → GitHub Actions** (Recommended)

**Current**: `.travis.yml` with Ubuntu Xenial  
**Target**: `.github/workflows/` with modern Ubuntu

**Benefits**:
- Travis CI free tier is limited
- GitHub Actions integrated with GitHub
- Better caching, faster builds
- More flexible

**Workflow to Create**:
```yaml
# .github/workflows/test.yml
- Test on Python 3.10, 3.11, 3.12
- Test on multiple OS: Ubuntu, macOS, Windows
- Build wheels for distribution
- Run linting, coverage
```

## Detailed Implementation Steps

### Immediate: Upgrade to NumPy 1.26 + Python 3.10

**Estimated Effort**: 2-4 hours

1. **Update version constraints**:
   - `setup.py`: python_requires, numpy version, classifiers
   - `requirements.txt`: numpy, scipy
   - `requirements-dev.txt`: ensure compatibility

2. **Update CI**:
   - Travis CI: update Python versions to 3.10, 3.11, 3.12
   - Or migrate to GitHub Actions

3. **Test thoroughly**:
   - Run all tests with new versions
   - Check for deprecation warnings
   - Fix any compatibility issues

4. **Update documentation**:
   - README.md badges
   - Installation instructions

### Short-term: Migrate to Meson Build System

**Estimated Effort**: 1-2 weeks (significant)

1. **Setup pyproject.toml**:
   - Define build dependencies (meson-python, Cython, etc.)
   - Define project metadata
   - Configure build backend

2. **Create meson.build files**:
   - Root: project setup, version, compiler flags
   - Each package: extension definitions
   - Handle C and Cython sources

3. **Handle templating**:
   - NumPy's `.c.src` templating won't work
   - Options:
     - Convert to Cython
     - Create custom preprocessing script
     - Write C directly

4. **Migrate extensions**:
   - Core C extensions
   - Cython extensions in csgraph
   - Sparsetools extensions

5. **Update tooling**:
   - Update `cythonize.py` or replace with meson's Cython support
   - Update wheel building scripts
   - Update Docker build files

6. **Test extensively**:
   - Build and install package
   - Run all tests
   - Check wheel building
   - Test on multiple platforms

### Long-term: Upgrade to NumPy 2.0

**Estimated Effort**: 1 week

1. **Review NumPy 2.0 migration guide**:
   - Read official migration documentation
   - Check C API changes affecting your code

2. **Update C extensions**:
   - Review NumPy C API usage
   - Update to NumPy 2.0 compatible APIs
   - Test compilation

3. **Update Cython code**:
   - Ensure Cython version supports NumPy 2.0
   - Update type declarations if needed

4. **Update version constraints**:
   - `pyproject.toml`: numpy>=2.0.0

5. **Test thoroughly**:
   - Complete test suite
   - Performance benchmarks
   - Memory usage checks

## Risk Assessment

### High Risk
- **Build system migration**: Complex, can break entire build
- **NumPy 2.0 C API changes**: Subtle bugs possible
- **Template system migration**: .c.src files need alternative

### Medium Risk
- **SciPy sparse API changes**: Generally stable but verify
- **Python version upgrade**: Should be straightforward
- **CI/CD migration**: Tooling change, not code

### Low Risk
- **Requirement version bumps**: If following phased approach
- **Documentation updates**: No code impact

## Testing Strategy

### Unit Tests
- Ensure all existing tests pass with new versions
- Add tests for any changed behavior
- Check edge cases affected by upgrades

### Integration Tests
- Test complete workflows
- Verify notebook examples still work
- Check command-line tools (`rbg` script)

### Performance Tests
- Benchmark key operations before/after
- Ensure no significant regressions
- Document any performance changes

### Compatibility Tests
- Test on multiple Python versions (3.10, 3.11, 3.12)
- Test on multiple platforms (Linux, macOS, Windows)
- Test with different NumPy/SciPy versions in supported range

## Timeline Estimate

### Conservative Approach (Recommended)

**Phase 1** (1 week):
- Upgrade to Python 3.10+, NumPy 1.26.x, SciPy 1.11+
- Update CI/CD
- Test and fix issues

**Phase 2** (2-3 weeks):
- Migrate to Meson build system
- Handle .c.src templating
- Extensive testing

**Phase 3** (1 week):
- Upgrade to NumPy 2.0
- Fix compatibility issues
- Final testing

**Total**: 4-5 weeks of focused work

### Aggressive Approach

**Phase 1** (3 days):
- Quick version bumps
- Basic testing

**Phase 2** (1-2 weeks):
- Rapid Meson migration
- Focus on getting it working

**Phase 3** (3 days):
- NumPy 2.0 upgrade
- Fix critical issues

**Total**: 2-3 weeks

## Rollback Plan

At each phase:
1. Create a git branch for the upgrade work
2. Keep main branch stable
3. Tag releases before major changes
4. Document known issues
5. Maintain compatibility shims if possible

If issues arise:
- Can stay on NumPy 1.26.x indefinitely (receives security updates)
- Can maintain separate branches for different versions
- Can vendor dependencies if needed

## Resources

### Documentation
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Meson Build System](https://mesonbuild.com/)
- [Meson Python](https://meson-python.readthedocs.io/)
- [SciPy Build Documentation](https://docs.scipy.org/doc/scipy/building/)

### Examples
- NumPy's own meson.build files
- SciPy's build system (migrated to Meson)
- Other projects: pandas, scikit-learn

## Recommendations

### Immediate Actions (This Week)
1. ✅ Create this upgrade plan
2. Create a feature branch: `feature/upgrade-numpy-scipy`
3. Update to Python 3.10+, NumPy 1.26.x, SciPy 1.11+
4. Get all tests passing
5. Create a release with these versions

### Next Steps (Next Month)
1. Study Meson build system
2. Create prototype meson.build for one extension
3. Gradually migrate all extensions
4. Test thoroughly on multiple platforms

### Future (3-6 months)
1. Migrate to NumPy 2.0 once Meson migration is stable
2. Consider modernizing other aspects:
   - Type hints (Python 3.10+ supports nice syntax)
   - Modern packaging (src layout?)
   - Better documentation (Sphinx + nbsphinx for notebooks)

### Don't Do (Yet)
1. Don't rush to NumPy 2.0 before Meson migration
2. Don't drop Python 3.9 support immediately (let users catch up)
3. Don't change APIs while upgrading (separate concerns)

## Conclusion

The upgrade path requires careful planning due to the **numpy.distutils deprecation**. The recommended approach is:

1. **Short-term**: Upgrade to NumPy 1.26.x, SciPy 1.11+, Python 3.10+ (relatively easy)
2. **Medium-term**: Migrate build system to Meson (challenging but necessary)
3. **Long-term**: Upgrade to NumPy 2.0 (moderate difficulty after Meson migration)

The most challenging aspect is the build system migration, which cannot be avoided for NumPy 2.0 support. However, NumPy 1.26.x provides a stable intermediate step.

**Total estimated effort**: 4-5 weeks for complete migration following the conservative approach.
