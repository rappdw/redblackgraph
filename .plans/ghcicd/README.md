# GitHub CI/CD Migration Documentation

This directory contains all documentation and resources for migrating from Travis CI and Codecov to GitHub Actions CI/CD.

## Documents

### 📋 [Migration Plan](./migration_plan.md)
Comprehensive plan covering strategy, timeline, and technical details for the migration.

**Key Sections:**
- Current state analysis
- Proposed GitHub Actions workflows
- 4-phase migration approach
- Coverage reporting options
- Security considerations
- Success criteria

### ✅ [Migration Checklist](./migration_checklist.md)
Detailed checklist to track migration progress through all phases.

**Usage:**
- Mark items complete as you progress
- Document issues encountered
- Track metrics comparison
- Record timeline

### 🔧 [Sample CI Workflow](./sample_ci_workflow.yml)
Example GitHub Actions workflow for continuous integration.

**Features:**
- Python 3.10, 3.11, 3.12 matrix
- Linting with pylint
- Testing with pytest
- Coverage with codecov
- Artifact uploads
- Caching strategy

**Deployment:**
```bash
# Copy to project .github/workflows directory
mkdir -p .github/workflows
cp .plans/ghcicd/sample_ci_workflow.yml .github/workflows/ci.yml
```

### 🚀 [Release Workflow](./.github/workflows/release.yml)
GitHub Actions workflow for releases and PyPI publishing.

### 📦 [Badge Setup Guide](./badge_setup_guide.md)
**Required for coverage badge to work!** Step-by-step guide to set up the GitHub Gist for coverage badges.

**Features:**
- Source distribution building
- Wheel building for multiple platforms
- Optional cibuildwheel integration
- GitHub Release creation
- PyPI publishing with trusted publishing

**Deployment:**
```bash
# Copy to project .github/workflows directory
cp .plans/ghcicd/sample_release_workflow.yml .github/workflows/release.yml
```

## Quick Start

### 1. Review the Migration Plan
Read through [migration_plan.md](./migration_plan.md) to understand:
- The migration strategy
- Timeline expectations
- Technical considerations
- Coverage reporting options

### 2. ✅ Coverage Solution Selected
**Decision**: GitHub Native Coverage (Option B)
- Codecov repository was deactivated
- Using GitHub native solution with Gist-based badges
- See [badge_setup_guide.md](./badge_setup_guide.md) for badge configuration

### 3. ✅ Workflows Implemented
Modify the sample workflows for your needs:
- Adjust Python versions if needed
- Configure coverage reporting choice
- Add or remove jobs as needed
- Customize caching strategies

### 4. Follow the Checklist
Use [migration_checklist.md](./migration_checklist.md) to:
- Track your progress
- Ensure no steps are missed
- Document decisions and issues
- Record metrics for comparison

## Migration Phases

### Phase 1: Setup (Week 1)
- Create GitHub Actions workflows
- Test on feature branch
- Validate functionality

### Phase 2: Parallel Operation (Weeks 2-5)
- Run both CI systems
- Monitor and compare results
- Build confidence in new system

### Phase 3: Cutover (Week 6)
- Make GitHub Actions primary
- Update badges and protection rules
- Disable Travis CI

### Phase 4: Cleanup (Week 7)
- Remove old configurations
- Optimize workflows
- Update documentation

## Key Files to Modify

During migration, these files will be affected:

```
Repository Root
├── .github/
│   └── workflows/
│       ├── ci.yml           (NEW - Phase 1)
│       ├── release.yml      (NEW - Phase 1)
│       └── integration.yml  (NEW - Phase 1, optional)
├── .travis.yml              (DEPRECATE - Phase 3, DELETE - Phase 4)
├── README.md                (UPDATE - Phase 3)
└── pyproject.toml          (POSSIBLY UPDATE - Phase 4)
```

## Resources

### GitHub Actions Documentation
- [GitHub Actions Overview](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Python Setup Action](https://github.com/actions/setup-python)
- [Cache Action](https://github.com/actions/cache)

### Coverage Tools
- [Codecov Action](https://github.com/codecov/codecov-action)
- [Coverage.py](https://coverage.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

### Build Tools
- [Meson Build](https://mesonbuild.com/)
- [cibuildwheel](https://cibuildwheel.readthedocs.io/)
- [PyPA Build](https://pypa-build.readthedocs.io/)

### Publishing
- [Trusted Publishing to PyPI](https://docs.pypi.org/trusted-publishers/)
- [PyPI Publish Action](https://github.com/pypa/gh-action-pypi-publish)

## Testing GitHub Actions Locally

You can test GitHub Actions workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI workflow locally
act -j test

# Run specific job
act -j lint

# Use specific platform
act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest
```

## Common Issues and Solutions

### Issue: Meson build fails in CI
**Solution**: Ensure all build dependencies are installed before pip install
```yaml
- name: Install build dependencies
  run: |
    pip install meson-python meson ninja cython
    pip install -e ".[test]"
```

### Issue: Coverage not uploading
**Solution**: Generate XML format for coverage tools
```yaml
- run: pytest --cov=redblackgraph --cov-report=xml
```

### Issue: Build too slow
**Solution**: Implement caching for pip and meson
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
```

## Getting Help

If you encounter issues during migration:

1. Check the sample workflows in this directory
2. Review GitHub Actions documentation
3. Check GitHub Actions community forums
4. Review similar Python projects using GitHub Actions
5. Consult the team or maintainers

## Status Dashboard

Track the overall migration status:

| Component | Status | Notes |
|-----------|--------|-------|
| Planning | ✅ Complete | All docs created |
| Phase 1: Setup | 🟡 In Progress | Workflows created, needs testing |
| Phase 2: Parallel | ⏸️ Not Started | |
| Phase 3: Cutover | ⏸️ Not Started | |
| Phase 4: Cleanup | ⏸️ Not Started | |

Legend:
- ✅ Complete
- 🟡 In Progress
- ⏸️ Not Started
- ❌ Blocked

---

**Last Updated**: 2025-10-22  
**Document Version**: 1.0  
**Maintainer**: TBD
