# Phase 1, Sprint 1: Architectural Clarification Questions

**Document Purpose**: Capture critical questions that must be answered before defining the detailed architecture and implementation plan for Sprint 1.

**Target Audience**: Project stakeholders, technical leads, and the engineering team

**Date**: 2025-10-21

---

## 1. Current State & Testing Infrastructure

### 1.1 Existing Test Coverage
- **Q1.1.1**: What is the current test coverage percentage? Do we have specific coverage requirements/thresholds we need to maintain? For this phase ensure that all tests are passing. we won't focus on increasing coverage
- **Q1.1.2**: Are there any known test failures in the current codebase on Python 3.6/3.7/3.8? If so, should these be documented as "expected failures" in the baseline? all test pass
- **Q1.1.3**: Do integration tests exist, or are we primarily working with unit tests? What about end-to-end tests? primarily unit tests
- **Q1.1.4**: Are there any platform-specific tests (Linux vs macOS vs Windows)? Which platforms must we support? Linux and macOS

### 1.2 CI/CD Pipeline
- **Q1.2.1**: Is Travis CI still the active CI/CD platform, or has the team migrated to GitHub Actions or another platform? Travis is still CI/CD platform. 
- **Q1.2.2**: Are there any private CI/CD runners or infrastructure we need access to for the migration work? No
- **Q1.2.3**: What is the current deployment/release process? Are there automated wheel builds on PyPI? No
- **Q1.2.4**: Do we have access to build multiple platforms (Linux, macOS, Windows) in CI? Which are required for Phase 1? Linux and macOS

### 1.3 Performance Benchmarking
- **Q1.3.1**: Are there existing performance benchmarks/tests, or do we need to create them from scratch? Don't worry about performance at this time.
- **Q1.3.2**: What are the critical performance-sensitive operations that must be benchmarked? (The sprint mentions matrix operations, sparse matrix operations, csgraph operations, and import times - is this complete?) Don't worry about performance at this time.
- **Q1.3.3**: What is an acceptable performance variance from the baseline? (+/- 5%? +/- 10%?) Don't worry about performance at this time.
- **Q1.3.4**: What hardware/environment should be used for performance benchmarking to ensure consistency? Don't worry about performance at this time.

---

## 2. Development Environment & Infrastructure

### 2.1 Python Version Management
- **Q2.1.1**: Is there a preferred tool for Python version management (pyenv, conda, asdf, etc.)? Any organizational standards? uv
- **Q2.1.2**: Should we use virtual environments (venv) or conda environments? Any preference? uv venv
- **Q2.1.3**: Where should the environments be located? In the project directory or in a standard system location? project directory
- **Q2.1.4**: Do we need to support ARM64/M1 Macs in addition to x86_64 architectures? apple silicon only

### 2.2 Build Dependencies
- **Q2.2.1**: What C/C++ compiler versions are required/supported? Are there minimum versions we need to enforce? gcc 11
- **Q2.2.2**: What version of Cython should we use during Phase 1? (Current setup.py lists it in setup_requires but without version constraints) cython 3.5
- **Q2.2.3**: Are there any system-level dependencies (BLAS/LAPACK libraries, etc.) that need to be documented? No
- **Q2.2.4**: What Fortran compiler is needed, if any? (The code checks for f90/f77 compilers) None

### 2.3 Development Machine Requirements
- **Q2.3.1**: What are the minimum hardware requirements for development machines (RAM, CPU cores, disk space)? 8GB RAM, 4 cores, 100GB disk space
- **Q2.3.2**: Are there any network/firewall considerations for accessing package repositories? None
- **Q2.3.3**: Do developers have sudo/admin access for installing Python versions and system dependencies? None

---

## 3. Code Architecture & Dependencies

### 3.1 Package Structure
- **Q3.1.1**: The project has 5 setup.py files. Are all of these still active and required, or are some legacy? all required
- **Q3.1.2**: Can you confirm the package hierarchy: `redblackgraph` → `core`, `sparse`, `reference`, `types`, `util`? Are there any planned changes to this structure? Not at this time
- **Q3.1.3**: Is the nested setup.py structure intentional for building separate extension modules, or is this a legacy pattern? Intentional
- **Q3.1.4**: Are there any plans to consolidate or refactor the package structure during this migration? If it makes sense in accomplishing the overall goal, then yes

### 3.2 C Extensions & Cython
- **Q3.2.1**: The 4 `.c.src` template files in `core/src/redblackgraph/` - are these all actively used? What functionality do they provide? Yes
- **Q3.2.2**: What is the relationship between the `.c.src` files and the `.pyx` files? Do they provide similar functionality in different ways? Yes
- **Q3.2.3**: Are the `reference` implementations still actively maintained, or are they purely for documentation/testing? Yes
- **Q3.2.4**: Do the C extensions have any external dependencies beyond NumPy/SciPy? No
- **Q3.2.5**: Are there any known issues with the current C extensions on newer Python versions? None

### 3.3 NumPy API Usage
- **Q3.3.1**: Are you aware of specific NumPy 1.20+ deprecations that your code uses? (e.g., `np.int`, `np.float`, old random API) No
- **Q3.3.2**: Does the codebase use NumPy's C API directly? If so, which version of the API? Yes, don't recall the version see if you can assess from source
- **Q3.3.3**: Are there any assumptions about NumPy's internal data structures that might break with NumPy 2.0 (even though Phase 1 only goes to 1.26)? Yes, we use einsum over internal numpy matrix representation

### 3.4 Dataclasses Usage
- **Q3.4.1**: How extensively are dataclasses used in the codebase? Are they in core functionality or peripheral utilities? Not extensively
- **Q3.4.2**: Are there any custom behaviors in the dataclasses that might behave differently between the backport and native implementation? No
- **Q3.4.3**: Can you provide examples of the most complex dataclass usage in the code? Not applicable

---

## 4. Git Workflow & Branching Strategy

### 4.1 Branching Model
- **Q4.1.1**: Is `main` (or `master`) the primary branch for releases, or is there a separate `develop` branch? Yes
- **Q4.1.2**: What is the typical merge strategy: merge commits, squash merges, or rebase? Merge commits
- **Q4.1.3**: Are there any long-lived feature branches currently active that we need to coordinate with? None
- **Q4.1.4**: How often is `main` released to PyPI? Is it every commit, manual releases, or something else? Manual releases

### 4.2 Code Review Process
- **Q4.2.1**: What is the code review process? How many approvals are required? no review
- **Q4.2.2**: Are there specific reviewers who must approve changes to C extensions vs Python code? no
- **Q4.2.3**: Are there any automated checks (linting, type checking, security scanning) that run on PRs? no
- **Q4.2.4**: What is the typical PR size (lines of code)? Should we aim for smaller, more frequent PRs during migration? not applicable

### 4.3 Versioning Strategy
- **Q4.3.1**: The project uses versioneer. Will Phase 1 completion trigger a major, minor, or patch version bump? major
- **Q4.3.2**: Do we need to maintain backward compatibility in the public API, or is this a breaking change release? breaking
- **Q4.3.3**: Should we use pre-release versions (alpha, beta, rc) during the Phase 1 branch development? no
- **Q4.3.4**: What is the CHANGELOG.md update process? not applicable

---

## 5. Testing Strategy & Acceptance Criteria

### 5.1 Test Execution
- **Q5.1.1**: What is the full command to run the test suite? (Travis CI shows `bin/test -u` - what does the `-u` flag do?) runs unit tests
- **Q5.1.2**: How long does the full test suite take to run on each Python version? under 5 minutes
- **Q5.1.3**: Are there any tests that are known to be flaky or environment-dependent? no
- **Q5.1.4**: What is the process for updating test expectations when behavior legitimately changes? not applicable

### 5.2 Acceptance Criteria Definition
- **Q5.2.1**: Who has final approval authority for declaring Sprint 1 complete? me
- **Q5.2.2**: Are there any external stakeholders (users, customers, partners) who should review the baseline documentation? no
- **Q5.2.3**: What level of detail is expected in the baseline reports? High-level summary or exhaustive detail? High-level summary
- **Q5.2.4**: Should the baseline capture information about test execution time, or just pass/fail status? pass/fail status

### 5.3 Known Issues & Technical Debt
- **Q5.3.1**: Are there any known issues in the current codebase that should be documented but not fixed in Phase 1? no
- **Q5.3.2**: Are there any workarounds or hacks in the current code that might become problems during migration? no
- **Q5.3.3**: Are there any pending security vulnerabilities that Phase 1 should address? no

---

## 6. Risk Management & Rollback

### 6.1 Risk Tolerance
- **Q6.1.1**: What is the risk tolerance for this migration? Is this a critical production system or a research project? research project
- **Q6.1.2**: What is the acceptable downtime if a rollback is needed? not applicable
- **Q6.1.3**: Are there any downstream projects or users that depend on specific behaviors that we must not break? no
- **Q6.1.4**: What is the communication plan if we discover a showstopper issue? not applicable

### 6.2 Backup & Rollback
- **Q6.2.1**: Do we need to maintain the ability to build wheels for Python 3.6/3.7/3.8 after Phase 1 completes? yes
- **Q6.2.2**: Should we tag a "pre-phase1" release point before starting development? no
- **Q6.2.3**: Is there a separate test PyPI instance we should use for pre-release testing? no
- **Q6.2.4**: What is the rollback procedure if Phase 1 needs to be abandoned? (Beyond just deleting the branch) not applicable

---

## 7. Documentation & Knowledge Transfer

### 7.1 Documentation Location
- **Q7.1.1**: Should all documentation go in the `.plans/` directory, or should some go in a `docs/` directory? all in .plans
- **Q7.1.2**: Is there an existing documentation site (ReadTheDocs, GitHub Pages, etc.) that needs to be updated? no
- **Q7.1.3**: Should the Jupyter notebooks be updated to reflect the new Python version requirements? yes
- **Q7.1.4**: Are there any API documentation tools (Sphinx, pdoc, etc.) that need to be run? no

### 7.2 Communication & Reporting
- **Q7.2.1**: Who should receive the daily standup updates and sprint reports? me
- **Q7.2.2**: What format should sprint demos take? Live demo, recorded video, written report? live demo
- **Q7.2.3**: Are there any specific stakeholders who need to be kept informed but not involved in day-to-day work? no
- **Q7.2.4**: Should we create a migration blog post or announcement when Phase 1 completes? no

### 7.3 Knowledge Preservation
- **Q7.3.1**: Are there any tribal knowledge or undocumented behaviors that should be captured during the baseline phase? no
- **Q7.3.2**: Should we create architecture decision records (ADRs) for significant decisions made during migration? yes
- **Q7.3.3**: Is there a need for training materials or migration guides for users/contributors? no

---

## 8. External Dependencies & Ecosystem

### 8.1 Dependency Constraints
- **Q8.1.1**: Are there any internal/proprietary packages that depend on redblackgraph? Do they have Python version constraints? no
- **Q8.1.2**: The `fs-crawler>=0.3.2` dependency - is this also being updated, or does it work with Python 3.10+? yes
- **Q8.1.3**: `XlsxWriter` and `scipy` - are there specific versions that are required or should be avoided? no
- **Q8.1.4**: Are there any optional dependencies (like the `numba` requirements file) that need to be considered? no

### 8.2 Ecosystem Compatibility
- **Q8.2.1**: Do any downstream projects pin to specific versions of redblackgraph? How will they be notified of the upgrade? no
- **Q8.2.2**: Are there any examples or tutorials in other repositories that reference redblackgraph installation? no
- **Q8.2.3**: What is the deprecation policy for old Python versions? Immediate drop or with warnings? immediate drop

---

## 9. Resource Allocation & Timeline

### 9.1 Team Availability
- **Q9.1.1**: Are the team members listed in the sprint plan (DevOps/Lead Engineer, QA Engineer, Performance Engineer) confirmed and available? yes
- **Q9.1.2**: What percentage of time will each person dedicate to this sprint? (Full-time vs part-time) full-time
- **Q9.1.3**: Are there any planned vacations or other commitments during the sprint period? no
- **Q9.1.4**: Who is the backup contact if the primary owner of a task is unavailable? me

### 9.2 Budget & Infrastructure Costs
- **Q9.2.1**: Are there any costs associated with CI/CD usage (GitHub Actions minutes, cloud runners)? no
- **Q9.2.2**: Do we need to purchase or license any tools for the migration? no
- **Q9.2.3**: Are there hardware costs for performance testing infrastructure? no

### 9.3 Timeline Flexibility
- **Q9.3.1**: The sprint is estimated at 2-3 days. Is there flexibility to extend if issues arise? yes
- **Q9.3.2**: Are there any hard deadlines driven by external factors (conferences, releases, fiscal year)? no
- **Q9.3.3**: What happens if Sprint 1 uncovers issues that require re-planning Sprints 2-4? no

---

## 10. Special Considerations

### 10.1 Library-Specific Concerns
- **Q10.1.1**: The `.c.src` template processing - do we have documentation on how numpy.distutils handles these files? no
- **Q10.1.2**: The versioneer integration - have there been any updates to versioneer that we should incorporate? unsure, but we should look into it
- **Q10.1.3**: The docker integration mentioned in README - should Docker images also be tested with new Python versions? yes
- **Q10.1.4**: What is the relationship between `redblackgraph` packages and potential conflicts with SciPy's internal csgraph? no

### 10.2 Future Phases Planning
- **Q10.2.1**: How detailed should our understanding of Phase 2 (Meson migration) be at this point? not applicable
- **Q10.2.2**: Are there any decisions in Sprint 1 that could make Phase 2 easier or harder? no
- **Q10.2.3**: Should we be collecting any specific information about the `.c.src` files to help with Meson migration planning? no
- **Q10.2.4**: Is there any tooling or infrastructure setup in Sprint 1 that will be reused in Phase 2? no

### 10.3 Compliance & Licensing
- **Q10.3.1**: The license is AGPLv3+ - do any dependencies have license conflicts when upgrading versions? no
- **Q10.3.2**: Are there any export compliance or security requirements for the build process? no
- **Q10.3.3**: Should we run any security scanning tools on the updated dependencies? no

---

## Summary & Next Steps

This document contains **78 specific questions** organized into **10 major categories**. These questions should be answered before finalizing the Sprint 1 implementation plan.

### Suggested Process

1. **Triage Questions** (30 min meeting):
   - Mark questions as: Critical (must answer before start), Important (answer in first 2 days), Optional (can defer)
   
2. **Research & Investigation** (1-2 days):
   - Technical lead investigates critical questions
   - Document answers in a companion document
   
3. **Architecture Review** (1-2 hour meeting):
   - Review answers with full team
   - Identify any new questions or concerns
   - Finalize Sprint 1 approach

4. **Implementation Planning** (Half day):
   - Create detailed task breakdown with answers incorporated
   - Assign specific responsibilities
   - Set up communication channels

### Priority Questions to Answer First

If time is limited, these questions should be answered before any work begins:

**Must-Have Answers:**
- Q1.2.1 (CI/CD platform)
- Q1.3.1 (Existing benchmarks)
- Q2.1.1 (Python version management tool)
- Q3.1.1 (Active setup.py files)
- Q4.1.1 (Primary branch)
- Q5.1.1 (Test execution command)
- Q9.1.1 (Team availability)

**Should-Have Answers (by end of Day 1):**
- Q1.1.2 (Known test failures)
- Q1.2.4 (Platform support requirements)
- Q3.2.1 (Active .c.src files)
- Q3.4.1 (Dataclasses usage extent)
- Q6.1.1 (Risk tolerance)
- Q7.1.1 (Documentation location)

---

**Document Status**: Ready for Review  
**Next Action**: Schedule clarification meeting with stakeholders  
**Owner**: Principal Architect / Tech Lead
