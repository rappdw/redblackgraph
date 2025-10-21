# Phase 1, Sprint 3: Code Modernization & Cleanup

**Sprint Goal**: Remove Python 3.6/3.7-specific code, eliminate dataclasses backport, address deprecation warnings, and modernize code for Python 3.10+ baseline.

**Duration**: 3-4 days  
**Team Size**: 2-3 engineers  
**Dependencies**: Sprint 2 complete (dependencies updated and building)

---

## Objectives

1. Remove dataclasses backport imports and conditionals
2. Address all NumPy 1.26 deprecation warnings
3. Update Python 3.6/3.7-specific code patterns
4. Modernize code to use Python 3.10+ features where beneficial
5. Update type hints to Python 3.10+ syntax (if applicable)
6. Ensure code quality standards maintained

---

## Tasks

### Task 3.1: Remove Dataclasses Backport
**Owner**: Software Engineer 1  
**Effort**: 4 hours

**Activities**:
- [ ] Search codebase for `from dataclasses import` with version conditionals
- [ ] Remove conditional imports (e.g., `if sys.version_info >= (3, 7)`)
- [ ] Replace backport imports with standard library imports
- [ ] Update any __init__.py files that conditionally import dataclasses
- [ ] Remove dataclasses from requirements files (already done in Sprint 2, verify)
- [ ] Run tests to ensure no breakage

**Search Pattern**:
```bash
grep -r "dataclasses" --include="*.py"
grep -r "python_version" --include="*.py"
grep -r "version_info" --include="*.py"
```

**Code Example**:
```python
# Before
import sys
if sys.version_info >= (3, 7):
    from dataclasses import dataclass
else:
    from dataclasses_backport import dataclass

# After (Python 3.10+ only)
from dataclasses import dataclass
```

**Files Expected to Change**:
- Any files using dataclass decorator
- Utility modules with version conditionals

**Acceptance Criteria**:
- No references to dataclasses backport
- No version conditionals for Python <3.10
- All tests pass
- Clean imports

---

### Task 3.2: Address NumPy 1.26 Deprecation Warnings
**Owner**: Software Engineer 2  
**Effort**: 8 hours

**Activities**:
- [ ] Run tests with deprecation warnings enabled: `pytest -W default`
- [ ] Categorize warnings by type and severity
- [ ] Review NumPy 1.26 migration guide for each warning
- [ ] Update code to use non-deprecated APIs
- [ ] Focus on high-priority deprecations:
  - numpy.ndarray type specifications
  - numpy.dtype usage
  - Array creation patterns
  - numpy.testing functions
- [ ] Document any warnings deferred to Phase 2/3
- [ ] Re-run tests to verify warnings resolved

**Deliverables**:
- `docs/numpy_deprecation_fixes.md` documenting:
  - Each warning addressed
  - Code changes made
  - Any deferred warnings with rationale

**Common Deprecations to Address**:
- `np.bool` → `np.bool_`
- `np.int` → `np.int_`
- Ragged array creation
- `np.matrix` usage (if any)

**Acceptance Criteria**:
- All critical deprecation warnings resolved
- No new test failures introduced
- Deferred warnings documented
- Code follows NumPy 1.26 best practices

---

### Task 3.3: Python 3.10+ Code Modernization
**Owner**: Software Engineer 1  
**Effort**: 6 hours

**Activities**:
- [ ] Update string formatting to use f-strings consistently
- [ ] Replace old-style type hints with Python 3.10+ union syntax (X | Y instead of Union[X, Y])
- [ ] Use structural pattern matching (match/case) where beneficial
- [ ] Update exception handling to use newer patterns
- [ ] Review and update any deprecated Python patterns
- [ ] Ensure consistent code style

**Code Patterns to Update**:

```python
# Type hints (if using)
# Before: Union[int, str]
# After: int | str

# Before: Optional[int]
# After: int | None

# String formatting consistency
# Prefer f-strings over .format() or %

# Modern dictionary merging (Python 3.9+)
# Before: {**dict1, **dict2}
# After: dict1 | dict2 (where appropriate)
```

**Files to Review**:
- Core module files
- Public API files
- Utility modules

**Acceptance Criteria**:
- Consistent modern Python 3.10+ idioms
- No deprecated Python patterns
- Code style consistent
- All tests pass

---

### Task 3.4: C Extension Code Review
**Owner**: C/Python Engineer  
**Effort**: 6 hours

**Activities**:
- [ ] Review C extension files for Python 3.10 compatibility
- [ ] Check for deprecated Python C API usage
- [ ] Update C code comments and documentation
- [ ] Verify NumPy C API usage is NumPy 1.26 compatible
- [ ] Check for memory leaks with new Python version
- [ ] Review error handling patterns

**Files to Review**:
- `redblackgraph/core/src/redblackgraph/redblackgraphmodule.c`
- `redblackgraph/core/src/redblackgraph/*.c.src`
- Any C template files

**Things to Check**:
- PyObject reference counting
- Python C API version compatibility
- NumPy C API calls
- Error handling and exceptions
- Memory allocation/deallocation

**Acceptance Criteria**:
- C extensions compile without warnings
- No deprecated C API usage
- Memory leak tests pass
- Documentation updated

---

### Task 3.5: Cython Code Updates
**Owner**: Cython Specialist  
**Effort**: 6 hours

**Activities**:
- [ ] Review all .pyx files in `sparse/csgraph/`
- [ ] Update Cython directives for Python 3.10+
- [ ] Check NumPy cimports for 1.26 compatibility
- [ ] Update type declarations if needed
- [ ] Verify Cython generates clean C code
- [ ] Test all Cython extensions

**Files to Review**:
- `redblackgraph/sparse/csgraph/_shortest_path.pyx`
- `redblackgraph/sparse/csgraph/_rbg_math.pyx`
- `redblackgraph/sparse/csgraph/_components.pyx`
- `redblackgraph/sparse/csgraph/_permutation.pyx`
- `redblackgraph/sparse/csgraph/_ordering.pyx`
- `redblackgraph/sparse/csgraph/_relational_composition.pyx`
- `redblackgraph/sparse/csgraph/_tools.pyx`

**Cython Directives to Review**:
```python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
```

**Acceptance Criteria**:
- All Cython files compile successfully
- Generated C code is clean
- NumPy integration works correctly
- All csgraph tests pass

---

### Task 3.6: Code Quality & Linting
**Owner**: QA Engineer / Software Engineer  
**Effort**: 4 hours

**Activities**:
- [ ] Run pylint with updated configuration
- [ ] Fix any new linting issues introduced
- [ ] Run black or other formatter (if used)
- [ ] Check for any TODO or FIXME comments related to Python version
- [ ] Update .pylintrc if needed for Python 3.10+
- [ ] Ensure code coverage maintained

**Commands**:
```bash
pylint redblackgraph/
black redblackgraph/ --check
pytest --cov=redblackgraph --cov-report=html
```

**Deliverables**:
- Updated `.pylintrc` (if needed)
- Code coverage report showing no regression

**Acceptance Criteria**:
- Linting passes or has documented exceptions
- Code formatting consistent
- No decrease in code coverage
- No new code smells introduced

---

### Task 3.7: Testing & Validation
**Owner**: QA Engineer  
**Effort**: 6 hours

**Activities**:
- [ ] Run full test suite on all Python versions (3.10, 3.11, 3.12)
- [ ] Verify no new test failures
- [ ] Check for any flaky tests
- [ ] Run tests with warnings enabled to verify no new warnings
- [ ] Perform smoke tests of key functionality
- [ ] Document any test updates needed

**Test Categories**:
- Unit tests: `pytest tests/`
- Integration tests
- C extension tests
- Cython extension tests
- CLI tool tests: `rbg --help`

**Acceptance Criteria**:
- All tests pass on Python 3.10, 3.11, 3.12
- No new warnings introduced
- Test execution time comparable to baseline
- No flaky tests

---

### Task 3.8: Documentation Updates
**Owner**: Tech Writer / Lead Engineer  
**Effort**: 3 hours

**Activities**:
- [ ] Update code comments referencing old Python versions
- [ ] Update docstrings if API changes
- [ ] Review README for any Python version references
- [ ] Update inline code examples
- [ ] Create migration notes for code changes

**Deliverables**:
- `docs/code_modernization_notes.md` documenting:
  - Major code changes
  - Breaking changes (if any)
  - Migration guidance for contributors

**Acceptance Criteria**:
- Documentation is accurate
- No references to Python <3.10
- Code examples work
- Migration notes comprehensive

---

## Sprint Deliverables Summary

1. **Modernized Codebase**: Python 3.10+ idioms throughout
2. **Clean Build**: No deprecation warnings from NumPy 1.26
3. **Updated Extensions**: C and Cython code compatible with new versions
4. **Code Quality**: Maintained or improved linting and coverage
5. **Documentation**: Updated to reflect code changes

---

## Acceptance Criteria (Sprint Level)

- [ ] No dataclasses backport references in codebase
- [ ] No Python version conditionals for <3.10
- [ ] All NumPy 1.26 deprecation warnings addressed or documented
- [ ] Code uses modern Python 3.10+ idioms consistently
- [ ] C extensions compile without warnings
- [ ] All Cython extensions build successfully
- [ ] Full test suite passes on Python 3.10, 3.11, 3.12
- [ ] No decrease in code coverage
- [ ] Linting passes
- [ ] Documentation updated
- [ ] Code review completed and approved

---

## Testing Requirements

### Deprecation Warning Tests
```bash
# Run with all warnings enabled
pytest -W default tests/
python -W default -m pytest tests/

# Check for specific deprecation warnings
pytest -W error::DeprecationWarning tests/
```

### Code Quality Tests
```bash
# Linting
pylint redblackgraph/

# Formatting (if using black)
black --check redblackgraph/

# Type checking (if using mypy)
mypy redblackgraph/

# Coverage
pytest --cov=redblackgraph --cov-report=html --cov-report=term
```

### Integration Tests
- Import all modules successfully
- Run example notebooks (if applicable)
- Test CLI tool functionality
- Verify wheel builds and installs

---

## Dependencies & Blockers

**Dependencies**: 
- Sprint 2 complete with dependencies updated
- Build working with NumPy 1.26 and SciPy 1.11

**Potential Blockers**:
- Extensive deprecation warnings requiring significant refactoring
- C/Cython code issues with new versions
- Test failures due to behavioral changes

**Mitigation**:
- Prioritize critical warnings over nice-to-have modernizations
- Have C/Python expert available for extension issues
- Document and defer non-critical items if needed

---

## Rollback Plan

If modernization causes issues:
1. Maintain separate commits for each major change category
2. Can selectively revert problematic changes
3. Keep original code in comments temporarily for reference
4. Document any rollbacks in sprint retrospective

**Commit Strategy**:
- Commit 1: Remove dataclasses backport
- Commit 2: Fix NumPy deprecations
- Commit 3: Python modernization
- Commit 4: C/Cython updates
- Commit 5: Code quality fixes

---

## Communication & Reporting

**Daily Standup Topics**:
- Deprecation warnings status
- Any surprising code issues discovered
- C/Cython build status
- Test results

**Sprint Review Demo**:
- Show clean test run with no warnings
- Demonstrate modernized code examples
- Present code coverage report
- Show before/after code comparisons

**Sprint Retrospective Focus**:
- Were deprecations harder than expected?
- Did we find any hidden Python version dependencies?
- Should we have scoped this differently?

---

## Risk Management

### High Risk Items
- **NumPy API changes**: May cause functional regressions
  - *Mitigation*: Thorough testing, reference NumPy docs

- **C extension issues**: Subtle bugs in C code
  - *Mitigation*: Memory leak testing, valgrind if needed

### Medium Risk Items
- **Test breakage**: Modernization may expose test issues
  - *Mitigation*: Run tests frequently, fix incrementally

- **Performance impact**: Code changes may affect performance
  - *Mitigation*: Run benchmarks from Sprint 1

---

## Next Sprint Preview

**Sprint 4** will focus on:
- Comprehensive testing across all Python versions
- Performance validation against baseline
- Edge case testing
- Wheel building verification
- Release candidate preparation

**Preparation for Sprint 4**:
- Ensure all code changes complete
- Review test coverage gaps
- Prepare test matrices
- Set up performance testing environment

---

## Code Review Guidelines

### Review Checklist
- [ ] No deprecated APIs used
- [ ] Python 3.10+ idioms used correctly
- [ ] Type hints updated (if applicable)
- [ ] Tests updated for code changes
- [ ] Documentation reflects changes
- [ ] No introduction of code smells
- [ ] Performance considerations documented

### Focus Areas
- NumPy API usage correctness
- C/Cython memory management
- Error handling patterns
- Test coverage for changed code

---

## Notes

- Prioritize functional correctness over cosmetic modernization
- Keep changes focused and atomic
- Document any "tech debt" to address in future phases
- Some optimizations may be better suited for Phase 2 or 3
- Ensure backward compatibility in public APIs (if applicable)

**Status**: Ready to start after Sprint 2 completion
