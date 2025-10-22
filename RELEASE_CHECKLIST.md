# Release Checklist for RedBlackGraph

Quick reference for releasing a new version to PyPI.

## Pre-Release

- [ ] All tests pass: `pytest`
- [ ] Code quality checks pass: `pylint redblackgraph`
- [ ] Documentation is up to date
- [ ] CHANGELOG updated with new version

## Version Management

**No manual version updates needed!** Version is automatically determined by `setuptools_scm` from git tags.

- [ ] Decide on version number (following semver: MAJOR.MINOR.PATCH)

## Local Testing

```bash
# Build wheels for current platform
./bin/build-wheels-cibuildwheel.sh

# Test wheel installation
pip install wheelhouse/redblackgraph-*.whl

# Verify import
python -c "import redblackgraph; print(redblackgraph.__version__)"

# Run tests
pytest
```

## Publish to Test PyPI (Optional but Recommended)

```bash
# Upload to Test PyPI
twine upload --repository testpypi wheelhouse/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ redblackgraph
```

## Release to Production PyPI

### Automated Method (Recommended)

```bash
# Commit any changes (changelog, docs, etc.)
git add CHANGELOG.md
git commit -m "Prepare release x.x.x"
git push origin main

# Create and push tag (this determines the version automatically)
git tag -a vx.x.x -m "Release version x.x.x"
git push origin vx.x.x
```

GitHub Actions will automatically:
1. Build wheels for all platforms (Linux, macOS, Windows)
2. Build source distribution
3. Publish to PyPI

### Manual Method

```bash
# Upload wheels
twine upload wheelhouse/*

# Or upload everything (wheels + sdist)
twine upload dist/*
```

## Post-Release Verification

- [ ] Check PyPI page: https://pypi.org/project/redblackgraph/
- [ ] Test installation: `pip install redblackgraph`
- [ ] Verify version: `python -c "import redblackgraph; print(redblackgraph.__version__)"`
- [ ] Check GitHub release is created
- [ ] Update documentation if needed

## Quick Commands Reference

```bash
# Build wheels locally (current platform)
./bin/build-wheels-cibuildwheel.sh

# Build source distribution
python -m build --sdist

# Upload to Test PyPI
twine upload --repository testpypi wheelhouse/*

# Upload to PyPI
twine upload wheelhouse/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ redblackgraph

# Install from PyPI
pip install redblackgraph
```

## Platform Support Matrix

| Platform | Architecture | Python Versions |
|----------|-------------|-----------------|
| Linux    | x86_64      | 3.10, 3.11, 3.12 |
| Linux    | aarch64     | 3.10, 3.11, 3.12 |
| macOS    | x86_64      | 3.10, 3.11, 3.12 |
| macOS    | arm64       | 3.10, 3.11, 3.12 |
| Windows  | x86_64      | 3.10, 3.11, 3.12 |

## Rollback Procedure

If a release has issues:

1. **Cannot delete from PyPI**, but you can:
   - Release a new patch version with fixes
   - Mark the version as "yanked" on PyPI (prevents new installs but doesn't break existing ones)

2. To yank a release:
   ```bash
   # Requires PyPI API token with appropriate permissions
   twine upload --repository pypi --skip-existing --yank "Reason for yanking" wheelhouse/*
   ```

## Version Numbering

RedBlackGraph uses **setuptools_scm** for automatic versioning from git tags.

### How It Works

- Git tag `v0.5.1` → Package version `0.5.1`
- Between releases → Version includes commit info (e.g., `0.5.1.post1+git.abc1234`)
- No tags → Fallback version `0.0.0.dev0`

### Semantic Versioning

RedBlackGraph follows [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 0.5.0)
  - **MAJOR**: Incompatible API changes
  - **MINOR**: New features, backward compatible
  - **PATCH**: Bug fixes, backward compatible

**Important:** Only the git tag determines the version. No need to update version in any source files.

## GitHub Actions Workflow

The workflow is triggered by:
1. **Automatic**: Pushing a tag starting with `v` (e.g., `v0.5.1`)
2. **Manual**: Go to Actions → "Build and Publish Wheels" → Run workflow
   - Choose: `none` (build only), `testpypi`, or `pypi`

## First-Time Setup

### PyPI Credentials

Choose ONE method:

**Method 1: Trusted Publishing (Recommended)**
- No tokens needed
- Configure on PyPI: Account Settings → Publishing → Add trusted publisher
- Repository: `rappdw/redblackgraph`
- Workflow: `build-wheels.yml`
- Environment: `pypi` (or `testpypi`)

**Method 2: API Tokens**
- Generate token on PyPI: Account Settings → API tokens
- Create `~/.pypirc`:
  ```ini
  [pypi]
  username = __token__
  password = pypi-YOUR-TOKEN-HERE
  ```

### Required Tools

```bash
pip install build twine cibuildwheel
```

## Contact

For issues with releases, contact: rappdw@gmail.com
