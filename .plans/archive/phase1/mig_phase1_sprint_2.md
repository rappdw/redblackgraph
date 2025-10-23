# Phase 1, Sprint 2: Dependency Updates

**Sprint Goal**: Update all dependency specifications to target Python 3.10+ with NumPy 1.26.x and SciPy 1.11+, ensuring the build system remains functional.

**Duration**: 3-4 days  
**Team Size**: 2-3 engineers  
**Dependencies**: Sprint 1 complete (baseline established)

---

## Objectives

1. Update setup.py across all packages to reflect new version requirements
2. Update requirements.txt and requirements-dev.txt files
3. Remove obsolete dependency conditionals (dataclasses backport)
4. Update Python version classifiers
5. Validate builds with new dependency specifications
6. Test installation process

---

## Tasks

### Task 2.1: Root setup.py Updates
**Owner**: Build Engineer  
**Effort**: 3 hours

**Activities**:
- [ ] Update `python_requires='>=3.10'`
- [ ] Update `install_requires` to:
  ```python
  'numpy>=1.26.0,<2.0.0',
  'scipy>=1.11.0',
  'XlsxWriter',
  'fs-crawler>=0.3.2',
  ```
- [ ] Update `setup_requires=['numpy>=1.26.0,<2.0.0']`
- [ ] Remove `dataclasses;python_version<"3.7"` from dependencies
- [ ] Update classifiers to remove Python 3.5-3.9, add 3.10, 3.11, 3.12
- [ ] Add comments explaining version constraints

**Files Modified**:
- `/setup.py`

**Acceptance Criteria**:
- setup.py reflects all new version requirements
- No references to Python versions below 3.10
- Version constraints properly specified with rationale comments

---

### Task 2.2: Nested setup.py Files Updates
**Owner**: Build Engineer  
**Effort**: 4 hours

**Activities**:
- [ ] Update `redblackgraph/setup.py` (if exists)
- [ ] Update `redblackgraph/core/setup.py` with NumPy dependency
- [ ] Update `redblackgraph/sparse/setup.py` with NumPy dependency
- [ ] Update `redblackgraph/sparse/csgraph/setup.py` with NumPy dependency
- [ ] Ensure consistency across all setup.py files
- [ ] Review numpy.distutils usage (keep for Phase 1)

**Files Modified**:
- `/redblackgraph/setup.py` (if exists)
- `/redblackgraph/core/setup.py`
- `/redblackgraph/sparse/setup.py`
- `/redblackgraph/sparse/csgraph/setup.py`

**Acceptance Criteria**:
- All setup.py files have consistent version requirements
- No conflicts between nested setup files
- numpy.distutils still functional

---

### Task 2.3: Requirements Files Updates
**Owner**: DevOps Engineer  
**Effort**: 2 hours

**Activities**:
- [ ] Update `requirements.txt`:
  ```
  numpy>=1.26.0,<2.0.0
  scipy>=1.11.0
  XlsxWriter
  fs-crawler>=0.3.2
  ```
- [ ] Remove `dataclasses; python_version < '3.7'` line
- [ ] Update `requirements-dev.txt` for compatibility
- [ ] Review `requirements-numba.txt` (document for later deprecation decision)
- [ ] Ensure all pinned versions are compatible

**Files Modified**:
- `/requirements.txt`
- `/requirements-dev.txt`

**Acceptance Criteria**:
- Requirements files match setup.py specifications
- No obsolete conditional dependencies
- Dev dependencies compatible with new versions

---

### Task 2.4: Classifiers and Metadata Updates
**Owner**: Build Engineer  
**Effort**: 2 hours

**Activities**:
- [ ] Review and update all classifiers in setup.py
- [ ] Remove classifiers:
  ```
  "Programming Language :: Python :: 3.5"
  "Programming Language :: Python :: 3.6"
  "Programming Language :: Python :: 3.7"
  "Programming Language :: Python :: 3.8"
  "Programming Language :: Python :: 3.9"
  ```
- [ ] Add classifiers:
  ```
  "Programming Language :: Python :: 3.10"
  "Programming Language :: Python :: 3.11"
  "Programming Language :: Python :: 3.12"
  ```
- [ ] Update any version-specific metadata
- [ ] Review and update package description if needed

**Files Modified**:
- `/setup.py`

**Acceptance Criteria**:
- Classifiers accurately reflect supported Python versions
- No references to unsupported versions
- PyPI metadata will be accurate

---

### Task 2.5: Build Validation with New Dependencies
**Owner**: QA Engineer  
**Effort**: 6 hours

**Activities**:
- [ ] Create clean virtual environments for Python 3.10, 3.11, 3.12
- [ ] Install updated dependencies in each environment
- [ ] Build C extensions: `python setup.py build_ext --inplace`
- [ ] Verify all extensions compile successfully
- [ ] Test import of all modules
- [ ] Document any build warnings or issues
- [ ] Compare build times with baseline

**Test Environments**:
- Python 3.10 + NumPy 1.26 + SciPy 1.11
- Python 3.11 + NumPy 1.26 + SciPy 1.11
- Python 3.12 + NumPy 1.26 + SciPy 1.11

**Acceptance Criteria**:
- Extensions build successfully in all environments
- No critical build errors or warnings
- Import tests pass
- Build times documented

---

### Task 2.6: Installation Testing
**Owner**: QA Engineer  
**Effort**: 4 hours

**Activities**:
- [ ] Test `pip install -e .` in clean environments
- [ ] Test `python setup.py develop` installation
- [ ] Verify all entry points work (rbg command)
- [ ] Test uninstall and reinstall process
- [ ] Document installation issues if any
- [ ] Create installation test script

**Deliverables**:
- `scripts/test_installation_sprint2.sh` - Automated installation test

**Acceptance Criteria**:
- Package installs successfully in all Python versions
- No dependency resolution conflicts
- Entry points accessible after installation

---

### Task 2.7: Wheel Building Verification
**Owner**: Build Engineer  
**Effort**: 4 hours

**Activities**:
- [ ] Update wheel building scripts if needed
- [ ] Build wheels: `python setup.py bdist_wheel`
- [ ] Verify wheel metadata (METADATA file)
- [ ] Test wheel installation in clean environment
- [ ] Compare wheel size with baseline
- [ ] Document wheel building process

**Test Commands**:
```bash
python setup.py bdist_wheel
pip install dist/RedBlackGraph-*.whl
python -c "import redblackgraph; print(redblackgraph.__version__)"
```

**Acceptance Criteria**:
- Wheels build successfully for all Python versions
- Wheel metadata reflects new requirements
- Wheels install and import successfully

---

### Task 2.8: Dependency Conflict Resolution
**Owner**: Lead Engineer  
**Effort**: 4 hours (buffer)

**Activities**:
- [ ] Test with minimum specified versions (NumPy 1.26.0, SciPy 1.11.0)
- [ ] Test with latest available versions
- [ ] Document any version conflicts discovered
- [ ] Update version constraints if needed
- [ ] Create dependency compatibility matrix

**Deliverables**:
- `docs/dependency_compatibility_matrix.md`

**Acceptance Criteria**:
- No unresolved dependency conflicts
- Compatibility matrix documents tested combinations
- Version ranges are appropriate

---

## Sprint Deliverables Summary

1. **Updated Dependency Files**: All setup.py and requirements.txt files updated
2. **Build Validation**: Successful builds across Python 3.10, 3.11, 3.12
3. **Installation Tests**: Automated installation testing
4. **Wheel Artifacts**: Successfully built wheels for all versions
5. **Compatibility Matrix**: Documented tested dependency combinations

---

## Acceptance Criteria (Sprint Level)

- [ ] All setup.py files updated with correct version constraints
- [ ] All requirements files updated and consistent
- [ ] Python classifiers updated (remove <3.10, add 3.10-3.12)
- [ ] Build succeeds in Python 3.10, 3.11, 3.12 with NumPy 1.26 + SciPy 1.11
- [ ] C extensions compile without critical errors
- [ ] Package installs successfully via pip
- [ ] Wheels build and install correctly
- [ ] No dependency resolution conflicts
- [ ] Installation test script created and passing
- [ ] Compatibility matrix documented

---

## Testing Requirements

### Build Tests
```bash
# For each Python version (3.10, 3.11, 3.12)
python -m venv test_env
source test_env/bin/activate
pip install --upgrade pip
pip install -e ".[test]"
python setup.py build_ext --inplace
pytest tests/ -v
deactivate
```

### Wheel Tests
```bash
python setup.py bdist_wheel
pip install dist/RedBlackGraph-*.whl --force-reinstall
python -c "import redblackgraph; print(redblackgraph.__version__)"
```

### Installation Tests
- Fresh environment installation
- Editable installation
- Wheel installation
- Uninstall and reinstall

---

## Dependencies & Blockers

**Dependencies**: 
- Sprint 1 complete with baseline established
- Access to Python 3.10, 3.11, 3.12

**Potential Blockers**:
- NumPy 1.26 or SciPy 1.11 availability issues
- Unexpected build failures with new dependencies
- C extension compilation issues

**Mitigation**:
- Have fallback version numbers ready
- Early testing in isolated environments
- Build engineer available for troubleshooting

---

## Rollback Plan

If critical issues arise:
1. Revert all dependency changes
2. Return to baseline (Sprint 1 state)
3. Analyze failure in isolated environment
4. Adjust version constraints
5. Retry with modified approach

**Rollback Command**:
```bash
git checkout phase1-upgrade
git revert <commit-range>
```

---

## Communication & Reporting

**Daily Standup Topics**:
- Build status in each Python version
- Any dependency conflicts discovered
- Installation test results

**Sprint Review Demo**:
- Show successful builds in all Python versions
- Demonstrate installation process
- Present compatibility matrix
- Show wheel artifacts

**Sprint Retrospective Focus**:
- Were version constraints appropriate?
- Any unexpected issues with NumPy 1.26?
- Build process smooth?

---

## Risk Management

### High Risk Items
- **NumPy 1.26 API changes**: May cause compilation issues
  - *Mitigation*: Early testing, review NumPy 1.26 release notes

- **SciPy sparse API changes**: May affect rb_matrix
  - *Mitigation*: Run existing tests early in sprint

### Medium Risk Items
- **Build system issues**: numpy.distutils may behave differently
  - *Mitigation*: Build engineer dedicated to this sprint

---

## Next Sprint Preview

**Sprint 3** will focus on:
- Removing dataclasses backport code
- Fixing deprecation warnings
- Code cleanup for Python 3.10+
- Updating type hints if applicable

**Preparation for Sprint 3**:
- Review code inventory from Sprint 1
- Understand dataclasses backport usage
- Familiarize with NumPy 1.26 deprecations

---

## Notes

- Keep setup.py files consistent across all packages
- Document any version constraint decisions
- NumPy <2.0 constraint is critical (numpy.distutils removed in 2.0)
- SciPy 1.11+ chosen for NumPy 1.26 compatibility
- Save wheel artifacts for comparison with later phases

**Status**: Ready to start after Sprint 1 completion
