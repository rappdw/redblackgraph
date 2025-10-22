# Phase 1, Sprint 1: Implementation Plan

**Sprint Goal**: Establish foundation for Phase 1 migration with dev environments, branches, and baseline documentation.

**Duration**: 2-3 days | **Team**: 1-2 engineers | **Date**: 2025-10-21

---

## Overview

This plan provides step-by-step implementation and QA procedures for Sprint 1 tasks, incorporating architectural clarifications.

**Key Decisions**:
- Use **uv** for Python version management
- Virtual environments in **project directory** (.venv-*)
- **Apple Silicon only** (ARM64)
- **gcc 11** and **Cython 3.5**
- **No performance benchmarking** (deferred)
- Documentation in **.plans/** directory
- **Create ADRs** for significant decisions

---

## Task 1.1: Development Environment Setup

**Owner**: DevOps/Lead Engineer | **Time**: 4 hours

### Implementation Steps

1. **Install uv and verify tools**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
gcc --version  # Should be gcc 11.x
```

2. **Install Python versions**:
```bash
cd /home/rappdw/dev/redblackgraph
uv python install 3.10 3.11 3.12
uv python list
```

3. **Create virtual environments**:
```bash
uv venv .venv-3.10 --python 3.10
uv venv .venv-3.11 --python 3.11
uv venv .venv-3.12 --python 3.12
```

4. **Install dependencies** (repeat for each version):
```bash
source .venv-3.10/bin/activate
pip install --upgrade pip setuptools wheel
pip install cython==3.5.*
pip install -e ".[dev,test]"
deactivate
```

5. **Verify builds**:
```bash
for v in 3.10 3.11 3.12; do
  source .venv-${v}/bin/activate
  python -c "import redblackgraph; print(redblackgraph.__version__)"
  python -c "import redblackgraph.core; import redblackgraph.sparse"
  deactivate
done
```

### Deliverable: `.plans/dev_setup.md`

Document should include:
- Prerequisites (uv, gcc 11, hardware requirements)
- Installation steps for uv and Python versions
- Virtual environment creation commands
- Dependency installation process
- Verification commands
- Troubleshooting guide
- IDE configuration notes

### QA Checklist
- [ ] Python 3.10, 3.11, 3.12 installed via uv
- [ ] Three .venv directories exist
- [ ] All dependencies install without errors
- [ ] `import redblackgraph` succeeds in all environments
- [ ] C extensions load in all environments
- [ ] Documentation complete and tested

---

## Task 1.2: Git Branch Strategy

**Owner**: Lead Engineer | **Time**: 2 hours

### Implementation Steps

1. **Branch already created**:
migration branch, currently checked out

2. **Create sprint completion tags** (at end of each sprint):
```bash
git tag -a phase1-sprint1-complete -m "Sprint 1 Complete"
git push origin phase1-sprint1-complete
```

### Deliverable: `.plans/branching_strategy.md`

Document should include:
- Branch structure diagram
- Workflow for daily development
- Commit message conventions
- Sprint milestone tagging process
- Rollback procedures
- Final merge-to-main process

### QA Checklist
- [ ] `phase1-upgrade` branch exists locally and remotely
- [ ] Can switch between main and phase1-upgrade
- [ ] Documentation explains branch workflow
- [ ] Rollback procedures documented
- [ ] Commit conventions defined

---

## Task 1.3: Current State Baseline Documentation

**Owner**: QA Engineer/Test Lead | **Time**: 6 hours

### Implementation Steps

1. **Setup baseline environments** (Python 3.6, 3.7, 3.8):
```bash
git checkout main
uv python install 3.6 3.7 3.8
for v in 3.6 3.7 3.8; do
  uv venv .venv-baseline-${v} --python ${v}
  source .venv-baseline-${v}/bin/activate
  pip install -e ".[dev,test]"
  deactivate
done
```

2. **Capture test results**:
```bash
# Create script
cat > .plans/capture_baseline.sh << 'SCRIPT'
#!/bin/bash
OUT=".plans/baseline_test_results.txt"
echo "Baseline Test Results - $(date)" > $OUT
echo "Commit: $(git rev-parse HEAD)" >> $OUT
for v in 3.6 3.7 3.8; do
  echo "\n=== Python ${v} ===" >> $OUT
  source .venv-baseline-${v}/bin/activate
  python --version >> $OUT
  pip list | grep -E "(numpy|scipy|cython)" >> $OUT
  bin/test -u 2>&1 | tee -a $OUT
  deactivate
done
SCRIPT
chmod +x .plans/capture_baseline.sh
./.plans/capture_baseline.sh
```

3. **Capture deprecation warnings**:
```bash
# Run tests with all warnings
for v in 3.6 3.7 3.8; do
  source .venv-baseline-${v}/bin/activate
  python -W all -m pytest tests/ 2>&1 | grep -i "warning\|deprecated" > .plans/baseline_warnings.txt
  deactivate
done
```

4. **Document build process**:
```bash
source .venv-baseline-3.8/bin/activate
python setup.py bdist_wheel
ls -lh dist/ > .plans/baseline_build_artifacts.txt
deactivate
```

### Deliverable: `.plans/phase1_baseline_report.md`

Report sections:
- **Executive Summary**: Test pass rates, Python versions, platforms
- **Test Results**: Table with pass/fail/duration for each Python version
- **Dependency Versions**: Current NumPy, SciPy, Cython versions
- **Deprecation Warnings**: List of warnings found
- **Build System**: Description of setup.py structure, C/Cython extensions
- **Code Inventory**: Dataclasses usage, NumPy API patterns
- **CI/CD**: Travis CI configuration
- **Recommendations**: Priorities for Phase 1 work

### QA Checklist
- [ ] Baseline environments created for Python 3.6-3.8
- [ ] Tests executed on all baseline versions
- [ ] Results captured in baseline_test_results.txt
- [ ] Warnings captured
- [ ] Build artifacts documented
- [ ] Baseline report complete with all sections
- [ ] Report includes dependency version matrix

---

## Task 1.4: Code Inventory and Analysis

**Owner**: Tech Lead | **Time**: 6 hours

### Implementation Steps

1. **Scan for dataclasses usage**:
```bash
grep -r "from dataclasses import" redblackgraph/ > .plans/dataclasses_usage.txt
grep -r "import dataclasses" redblackgraph/ >> .plans/dataclasses_usage.txt
```

2. **Identify NumPy deprecations**:
```bash
# Scan for common deprecated patterns
grep -r "np\.int[^a-z]" redblackgraph/ > .plans/numpy_deprecated_types.txt
grep -r "np\.float[^a-z]" redblackgraph/ >> .plans/numpy_deprecated_types.txt
grep -r "np\.bool[^a-z]" redblackgraph/ >> .plans/numpy_deprecated_types.txt
grep -r "numpy\.random\.RandomState" redblackgraph/ >> .plans/numpy_deprecated_types.txt
```

3. **Review all setup.py files**:
```bash
find . -name "setup.py" -type f > .plans/setup_files_list.txt
```

4. **Inspect C API usage in .c.src files**:
```bash
# Look for specific C API patterns
for file in redblackgraph/core/src/redblackgraph/*.c.src; do
  echo "=== $file ===" >> .plans/c_api_patterns.txt
  grep -E "PyArray|NPY_|npy_" $file >> .plans/c_api_patterns.txt
done
```

5. **Check versioneer**:
```bash
# Compare current versioneer with latest
pip list | grep versioneer > .plans/current_versioneer.txt
```

### Deliverable: `.plans/phase1_code_inventory.md`

Document sections:
- **Dataclasses Usage**: Files, usage patterns, complexity assessment
- **NumPy Deprecated Patterns**: List of files using np.int, np.float, etc.
- **Setup.py Files**: All 5 files with their purposes
- **C Extensions**: .c.src template files and their functionality
- **Cython Extensions**: .pyx files and their functionality
- **Python Version Conditionals**: Any version-specific code
- **NumPy C API**: Patterns found, potential compatibility issues
- **Versioneer Status**: Current vs latest version

### Modification Checklist (`.plans/phase1_modification_checklist.md`):
```markdown
# Phase 1 Modification Checklist

## Sprint 2: Dependency Updates
- [ ] Update setup.py: Python version classifiers
- [ ] Update setup.py: Remove dataclasses conditional dependency
- [ ] Update setup.py: Set numpy>=1.26.0, scipy>=1.11.0
- [ ] Update setup.py: Set cython>=3.5.0
- [ ] Update requirements.txt: Remove dataclasses line
- [ ] Update .travis.yml: Change Python versions to 3.10, 3.11, 3.12
- [ ] All 5 setup.py files reviewed and updated

## Sprint 3: Code Modernization
- [ ] Remove dataclasses backport imports
- [ ] Replace np.int with np.int_ or int
- [ ] Replace np.float with np.float_ or float
- [ ] Replace np.bool with np.bool_ or bool
- [ ] Update any deprecated NumPy C API calls
- [ ] Test all C extensions compile with NumPy 1.26
- [ ] Test all Cython extensions compile with Cython 3.5
- [ ] Verify einsum usage still works

## Sprint 4: Testing & Validation
- [ ] All tests pass on Python 3.10, 3.11, 3.12
- [ ] No deprecation warnings
- [ ] Builds succeed on Linux and macOS
- [ ] Docker images updated and tested
- [ ] Jupyter notebooks updated
```

### QA Checklist
- [ ] Dataclasses usage documented
- [ ] NumPy deprecated patterns identified
- [ ] All setup.py files listed with purposes
- [ ] C/Cython extensions catalogued
- [ ] NumPy C API patterns documented
- [ ] Versioneer version checked
- [ ] Modification checklist created
- [ ] No surprises - comprehensive inventory

---

## Task 1.5: Risk Assessment and Mitigation Plan

**Owner**: Tech Lead/PM | **Time**: 4 hours

### Implementation Steps

1. **Identify technical risks**:
   - NumPy C API compatibility with NumPy 1.26
   - Cython 3.5 compatibility issues
   - Breaking changes in dependency behavior
   - Build failures on target platforms
   - Test failures after migration

2. **Document rollback procedures**:
   - Branch deletion process
   - Maintaining old Python 3.6-3.8 wheels
   - Version tag strategy

3. **Create contingency plans**:
   - For each identified risk, document mitigation
   - Include detection methods and resolution steps

### Deliverable: `.plans/phase1_risk_mitigation.md`

Document sections:
- **Risk Summary Table**: Risk name, severity (H/M/L), likelihood, mitigation
- **Technical Risks**:
  - NumPy C API changes
  - Cython compatibility
  - Platform-specific build failures
  - Test failures
  - Performance regressions
- **Mitigation Strategies**: For each risk
- **Detection Methods**: How to identify each risk early
- **Rollback Procedure**: Step-by-step emergency rollback
- **Communication Plan**: Who to contact, escalation path
- **Success Criteria**: Clear go/no-go decision criteria

Example risk table:
| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| NumPy C API incompatibility | High | Medium | Audit C code early; test with NumPy 1.26 |
| Cython 3.5 breaks extensions | High | Low | Test compilation in Sprint 2 |
| Platform build failures | Medium | Medium | Test on both macOS and Linux |

### QA Checklist
- [ ] All major technical risks identified
- [ ] Each risk has severity and likelihood
- [ ] Mitigation strategy for each risk
- [ ] Rollback procedure clearly documented
- [ ] Communication/escalation plan defined
- [ ] Success criteria established

---

## Sprint Completion Criteria

**Sprint 1 is complete when**:
- [ ] All three Python environments (3.10, 3.11, 3.12) working
- [ ] `phase1-upgrade` branch created and pushed
- [ ] Baseline test results documented for Python 3.6-3.8
- [ ] Code inventory complete with modification checklist
- [ ] Risk assessment documented
- [ ] All documentation in `.plans/` directory
- [ ] Sprint demo prepared (showing environments, baseline report, inventory)

---

## Deliverables Summary

| Deliverable | Location | Owner |
|------------|----------|-------|
| Dev setup guide | `.plans/dev_setup.md` | DevOps/Lead |
| Branching strategy | `.plans/branching_strategy.md` | Lead Engineer |
| Baseline report | `.plans/phase1_baseline_report.md` | QA/Test Lead |
| Test results | `.plans/baseline_test_results.txt` | QA/Test Lead |
| Warnings log | `.plans/baseline_warnings.txt` | QA/Test Lead |
| Code inventory | `.plans/phase1_code_inventory.md` | Tech Lead |
| Modification checklist | `.plans/phase1_modification_checklist.md` | Tech Lead |
| Risk assessment | `.plans/phase1_risk_mitigation.md` | Tech Lead/PM |

---

## Troubleshooting Guide

### Environment Issues

**Problem**: uv not found
```bash
# Solution: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

**Problem**: Build fails with "gcc not found"
```bash
# Solution: Install gcc 11
# macOS: brew install gcc@11
# Linux: apt install gcc-11
```

**Problem**: Cython compilation fails
```bash
# Solution: Ensure Cython 3.5 installed
pip list | grep -i cython
pip install cython==3.5.* --force-reinstall
```

### Test Execution Issues

**Problem**: `bin/test -u` fails with import errors
```bash
# Solution: Verify installation
pip show RedBlackGraph
pip install -e ".[dev,test]" --force-reinstall
```

**Problem**: Tests pass on 3.10 but fail on 3.11/3.12
```bash
# Solution: Check for Python version-specific issues
python -W all -m pytest tests/ -v  # Run with warnings
```

### Git/Branch Issues

**Problem**: Cannot push to phase1-upgrade
```bash
# Solution: Verify remote branch
git remote -v
git push -u origin phase1-upgrade
```

### Baseline Capture Issues

**Problem**: Python 3.6/3.7 not available
```bash
# Solution: Install with uv
uv python install 3.6 3.7
```

**Problem**: Tests hang or timeout
```bash
# Solution: Run with timeout
timeout 10m bin/test -u
```

---

## Architecture Decision Records

Create ADRs for significant decisions made during Sprint 1:

### ADR-001: Use uv for Python Version Management

**Status**: Accepted  
**Context**: Need reliable Python version management for 3.10, 3.11, 3.12  
**Decision**: Use `uv` instead of pyenv/conda  
**Consequences**: 
- Faster environment creation
- Better dependency resolution
- Requires team to learn uv tooling

### ADR-002: No Performance Benchmarking in Sprint 1

**Status**: Accepted  
**Context**: Sprint 1 focuses on foundation, performance deferred  
**Decision**: Skip performance baseline capture  
**Consequences**:
- Faster Sprint 1 completion
- Performance validation deferred to later sprints
- May discover performance issues later

### ADR-003: Apple Silicon Only

**Status**: Accepted  
**Context**: Development hardware standardization  
**Decision**: Support only ARM64/Apple Silicon for development  
**Consequences**:
- Simplified development setup
- CI/CD still tests x86_64 (Linux)
- Developers need Apple Silicon Macs

Create file: `.plans/architecture_decisions.md` documenting all ADRs.

---

## Next Steps After Sprint 1

**Sprint 2 Preparation**:
1. Review `.plans/phase1_modification_checklist.md`
2. Ensure understanding of dependency version constraints
3. Review NumPy 1.26 and SciPy 1.11 release notes
4. Identify any additional deprecated patterns from baseline warnings

**Sprint 2 Focus**:
- Update all 5 setup.py files
- Update requirements.txt
- Update .travis.yml
- Test builds on Python 3.10, 3.11, 3.12

---

**Document Status**: Ready for Execution  
**Owner**: Engineering Team  
**Approver**: Project Owner (Daniel Rapp)  
**Last Updated**: 2025-10-21
