# GitHub CI/CD Migration Plan

## Executive Summary

This document outlines the plan to migrate from Travis CI and Codecov to GitHub Actions for continuous integration and code coverage reporting.

## Current State

### Travis CI Configuration
- **File**: `.travis.yml`
- **Python Versions**: 3.10, 3.11, 3.12
- **OS**: Ubuntu Jammy (22.04)
- **Build System**: Meson + Ninja + Cython
- **Test Runner**: pytest with coverage
- **Coverage**: Codecov integration

### Current Workflow
1. Install build dependencies (meson-python, meson, ninja, cython)
2. Install project with test dependencies: `pip install -e ".[test]"`
3. Run tests: `bin/test -u`
4. Upload coverage to Codecov

### Dependencies
- Meson >= 1.2.0
- meson-python >= 0.15.0
- Ninja
- Cython >= 3.0
- NumPy >= 1.26.0, < 2.0
- pytest, pytest-cov, pylint (test dependencies)

## Migration Goals

1. **Replace Travis CI** with GitHub Actions for CI/CD
2. **Replace Codecov** with GitHub's native code coverage or alternative
3. **Maintain test coverage** requirements (65% minimum)
4. **Support multiple Python versions** (3.10, 3.11, 3.12)
5. **Update badges** in README.md
6. **Add additional workflows** for improved automation

## Proposed GitHub Actions Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

**Triggers**:
- Push to main/master branch
- Pull requests to main/master
- Manual workflow dispatch

**Jobs**:
- **Lint**: Run pylint on Python 3.12
- **Test**: Matrix build across Python 3.10, 3.11, 3.12
  - Build with Meson
  - Run unit tests with coverage
  - Upload coverage artifacts

**Coverage Options**:
- Option A: Use Codecov GitHub Action (easier migration)
- Option B: Use Coverage.py with GitHub's code coverage
- Option C: Use Coveralls

### 2. Release Workflow (`.github/workflows/release.yml`)

**Triggers**:
- Tag push matching `v*.*.*`
- Manual workflow dispatch

**Jobs**:
- Build wheels for multiple platforms using cibuildwheel
- Build source distribution
- Publish to PyPI (requires PyPI token)

### 3. Integration Test Workflow (`.github/workflows/integration.yml`)

**Triggers**:
- Scheduled (e.g., weekly)
- Manual workflow dispatch

**Jobs**:
- Run integration tests: `bin/test -i`

### 4. Documentation Workflow (optional)

**Triggers**:
- Push to main branch
- Manual workflow dispatch

**Jobs**:
- Generate Jupyter notebook PDFs
- Deploy to GitHub Pages

## Migration Steps

### Phase 1: Setup GitHub Actions (No Breaking Changes)

1. **Create `.github/workflows/` directory structure**
   ```
   .github/
   └── workflows/
       ├── ci.yml
       ├── release.yml
       └── integration.yml
   ```

2. **Implement CI workflow**
   - Configure matrix strategy for Python versions
   - Set up Meson build environment
   - Run tests with coverage
   - Configure coverage reporting

3. **Test workflows**
   - Push to a feature branch
   - Verify all jobs pass
   - Compare coverage reports with Travis CI

4. **Update documentation**
   - Add GitHub Actions badges to README.md (alongside existing badges)
   - Document workflow structure in `.plans/ghcicd/`

### Phase 2: Parallel Operation

1. **Run both CI systems simultaneously**
   - Travis CI continues as primary
   - GitHub Actions runs in parallel
   - Compare results and coverage metrics

2. **Monitor for issues**
   - Check for test flakiness
   - Verify coverage accuracy
   - Ensure build times are acceptable

3. **Duration**: 2-4 weeks or 10-20 commits

### Phase 3: Cutover

1. **Update README badges**
   - Replace Travis CI badge with GitHub Actions
   - Replace Codecov badge (if using alternative)

2. **Disable Travis CI**
   - Keep `.travis.yml` for reference
   - Add comment indicating deprecation

3. **Make GitHub Actions required**
   - Configure branch protection rules
   - Require CI checks to pass before merge

### Phase 4: Cleanup

1. **Remove Travis CI configuration**
   - Delete or archive `.travis.yml`
   - Remove Travis CI specific configurations

2. **Optimize workflows**
   - Add caching for pip dependencies
   - Add caching for Meson build artifacts
   - Fine-tune matrix strategy if needed

3. **Document new process**
   - Update contributor guidelines
   - Document local testing procedures

## Technical Considerations

### Build System Compatibility

Meson build system works well with GitHub Actions:
- Native support for pip installation
- Build artifacts can be cached
- Parallel builds possible with ninja

### Coverage Reporting Options

#### Option A: Continue with Codecov (Recommended for Migration)
**Pros**:
- Minimal changes to workflow
- Familiar UI and features
- Easy integration via codecov/codecov-action@v3

**Cons**:
- External dependency
- Requires signup/token management

**Implementation**:
```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    fail_ci_if_error: true
```

#### Option B: GitHub Native (Code Coverage Summary)
**Pros**:
- No external dependencies
- Built into GitHub
- Free for public repositories

**Cons**:
- Less feature-rich than Codecov
- Requires manual setup

**Implementation**:
```yaml
- name: Generate coverage report
  run: pytest --cov=redblackgraph --cov-report=xml

- name: Coverage Summary
  uses: irongut/CodeCoverageSummary@v1.3.0
  with:
    filename: coverage.xml
```

#### Option C: Coveralls
**Pros**:
- Popular alternative to Codecov
- Similar feature set

**Cons**:
- Another external service

### Caching Strategy

Implement caching to speed up builds:

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache Meson build
  uses: actions/cache@v3
  with:
    path: builddir
    key: ${{ runner.os }}-meson-${{ hashFiles('meson.build') }}
```

### Matrix Strategy

```yaml
strategy:
  matrix:
    os: [ubuntu-latest]
    python-version: ['3.10', '3.11', '3.12']
  fail-fast: false
```

## Security Considerations

1. **Secrets Management**
   - Store PyPI token as GitHub secret
   - Use GitHub's OIDC provider for trusted publishing
   - Rotate tokens periodically

2. **Dependency Scanning**
   - Enable Dependabot
   - Set up security advisories
   - Configure automated dependency updates

3. **Code Scanning**
   - Enable CodeQL analysis
   - Configure SARIF uploads

## Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 1: Setup | 1 week | GitHub Actions workflows created and tested |
| Phase 2: Parallel | 2-4 weeks | Both systems running, confidence built |
| Phase 3: Cutover | 1 week | GitHub Actions primary, Travis disabled |
| Phase 4: Cleanup | 1 week | Travis removed, workflows optimized |
| **Total** | **5-7 weeks** | Full migration complete |

## Success Criteria

- [ ] All tests passing on GitHub Actions
- [ ] Coverage reports matching Travis CI baseline (≥65%)
- [ ] Build times comparable or better than Travis CI
- [ ] Badges updated in README
- [ ] Documentation updated
- [ ] No regression in test reliability
- [ ] Team comfortable with new workflow

## Rollback Plan

If issues arise during migration:

1. **Phase 1-2**: Continue using Travis CI as primary
2. **Phase 3**: Re-enable Travis CI in branch protection
3. **Phase 4**: Restore `.travis.yml` if needed

Keep `.travis.yml` for at least one release cycle after cutover.

## Additional Improvements

### Opportunities with GitHub Actions

1. **Pre-commit Hooks Integration**
   - Run black formatter
   - Run mypy type checking
   - Run security linters (bandit, safety)

2. **Multi-platform Builds**
   - Add macOS and Windows to matrix
   - Test cross-platform compatibility

3. **Wheel Building**
   - Use cibuildwheel for binary wheels
   - Support multiple architectures (x86_64, aarch64)

4. **Automated Releases**
   - Automatic changelog generation
   - GitHub release notes from commits
   - Automated PyPI publishing

5. **Performance Benchmarks**
   - Run benchmark suite
   - Track performance over time
   - Detect performance regressions

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Meson Build System](https://mesonbuild.com/)
- [Codecov GitHub Action](https://github.com/codecov/codecov-action)
- [cibuildwheel Documentation](https://cibuildwheel.readthedocs.io/)
- [GitHub's OIDC for PyPI](https://docs.pypi.org/trusted-publishers/)

## Appendix: Example Badge Updates

### Current (README.md)
```markdown
[![TravisCI](https://api.travis-ci.org/rappdw/redblackgraph.svg?branch=master)](https://travis-ci.org/rappdw/redblackgraph)
[![Coverage](https://codecov.io/gh/rappdw/redblackgraph/branch/master/graph/badge.svg)](https://codecov.io/gh/rappdw/redblackgraph)
```

### Proposed (with Codecov)
```markdown
[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/rappdw/redblackgraph/branch/master/graph/badge.svg)](https://codecov.io/gh/rappdw/redblackgraph)
```

### Proposed (with GitHub native coverage)
```markdown
[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rappdw/GIST_ID/raw/coverage.json)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
```

## Next Steps

1. Review this migration plan with team
2. Choose coverage reporting solution
3. Create feature branch for GitHub Actions setup
4. Implement Phase 1 workflows
5. Begin parallel testing period

---

**Document Version**: 1.0  
**Created**: 2025-10-22  
**Author**: Cascade  
**Status**: Draft - Pending Review
