# PyPI Publishing Guide for RedBlackGraph

This document describes how to build and publish wheels for RedBlackGraph to PyPI.

## Overview

RedBlackGraph uses `cibuildwheel` to build wheels for multiple platforms and Python versions:
- **Python versions**: 3.10, 3.11, 3.12
- **Platforms**: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
- **Build system**: meson-python

## Prerequisites

### For Local Building

1. Install cibuildwheel:
   ```bash
   pip install cibuildwheel
   ```

2. Install twine for uploading:
   ```bash
   pip install twine
   ```

3. Set up PyPI credentials:
   - Create an account on [PyPI](https://pypi.org) and [Test PyPI](https://test.pypi.org)
   - Generate API tokens from account settings
   - Configure in `~/.pypirc`:
     ```ini
     [distutils]
     index-servers =
         pypi
         testpypi

     [pypi]
     username = __token__
     password = pypi-<your-token-here>

     [testpypi]
     repository = https://test.pypi.org/legacy/
     username = __token__
     password = pypi-<your-token-here>
     ```

### For GitHub Actions (Recommended)

1. Set up **Trusted Publishing** on PyPI (no API tokens needed):
   - Go to PyPI project settings → Publishing
   - Add GitHub as a trusted publisher
   - Provide: `rappdw/redblackgraph` and workflow: `build-wheels.yml`

2. Configure GitHub repository settings:
   - Go to Settings → Environments
   - Create environments: `pypi` and `testpypi`
   - Add required reviewers if desired

## Building Wheels Locally

### Build for Current Platform

Use the provided script to build wheels for your current platform:

```bash
./bin/build-wheels-cibuildwheel.sh
```

This will:
- Install cibuildwheel if needed
- Clean previous builds
- Build wheels for the current platform
- Place wheels in `./wheelhouse/`

### Build for Specific Platform

```bash
# Linux
cibuildwheel --platform linux --output-dir wheelhouse

# macOS
cibuildwheel --platform macos --output-dir wheelhouse

# Windows
cibuildwheel --platform windows --output-dir wheelhouse
```

### Test Wheels Locally

```bash
# Install the wheel
pip install wheelhouse/redblackgraph-*.whl

# Test import
python -c "import redblackgraph; print(redblackgraph.__version__)"

# Run tests
pytest
```

## Publishing to PyPI

### Method 1: Automated via GitHub Actions (Recommended)

#### For a Release

1. Prepare release (update changelog, documentation, etc.)
2. Commit and push changes:
   ```bash
   git commit -am "Prepare release 0.5.1"
   git push origin main
   ```
3. Create and push a version tag:
   ```bash
   git tag -a v0.5.1 -m "Release version 0.5.1"
   git push origin v0.5.1
   ```
4. GitHub Actions will automatically:
   - Determine version from the git tag (via setuptools_scm)
   - Build wheels for all platforms
   - Build source distribution
   - Publish to PyPI

**Note**: The version is automatically determined from the git tag. Do not manually update version numbers in any files.

#### Manual Trigger

You can also manually trigger the workflow:

1. Go to GitHub Actions → "Build and Publish Wheels"
2. Click "Run workflow"
3. Select target:
   - `none`: Build only, don't publish
   - `testpypi`: Build and publish to Test PyPI
   - `pypi`: Build and publish to production PyPI

### Method 2: Manual Upload

If you've built wheels locally and want to upload manually:

#### Upload to Test PyPI (for testing)

```bash
twine upload --repository testpypi wheelhouse/*
```

Test the installation:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ redblackgraph
```

Note: `--extra-index-url` is needed because Test PyPI doesn't have all dependencies.

#### Upload to Production PyPI

```bash
twine upload wheelhouse/*
```

Or specify the repository explicitly:
```bash
twine upload --repository pypi wheelhouse/*
```

## Building Source Distribution

If you only want to build a source distribution (sdist):

```bash
# Install build tool
pip install build

# Build sdist
python -m build --sdist

# The .tar.gz will be in dist/
```

Upload sdist:
```bash
twine upload dist/redblackgraph-*.tar.gz
```

## Version Management

### Automatic Versioning with setuptools_scm

RedBlackGraph uses `setuptools_scm` to **automatically determine version from git tags**. You don't need to manually update version numbers in any files.

When releasing a new version:

1. Update changelog/release notes

2. Commit changes:
   ```bash
   git commit -am "Prepare release 0.5.1"
   ```

3. Tag the release:
   ```bash
   git tag -a v0.5.1 -m "Release version 0.5.1"
   git push origin main
   git push origin v0.5.1
   ```

The version is automatically extracted from the git tag:
- **Tag `v0.5.1`** → Package version `0.5.1`
- **Between tags** → Version includes commit info (e.g., `0.5.1.post1+git.abc1234`)
- **No tags** → Version `0.0.0.dev0`

**Note:** The version in `meson.build` is a fallback for meson's internal use and does not affect the package version.

## Verification Checklist

Before publishing a release, verify:

- [ ] All tests pass locally: `pytest`
- [ ] Wheels build successfully for current platform: `./bin/build-wheels-cibuildwheel.sh`
- [ ] Test wheel installs and imports correctly
- [ ] Changelog updated
- [ ] Git tag created with correct version number (e.g., `v0.5.1`)
- [ ] Test PyPI upload works (optional but recommended)

## Troubleshooting

### Build Failures

**Issue**: Meson can't find NumPy headers
- **Solution**: Ensure NumPy is installed in the build environment (handled automatically by cibuildwheel)

**Issue**: Ninja not found
- **Solution**: The cibuildwheel config installs ninja automatically. If building manually, install: `pip install ninja`

**Issue**: Wrong Python version
- **Solution**: RedBlackGraph requires Python ≥3.10. Check: `python --version`

### Upload Failures

**Issue**: "403 Forbidden" or "Invalid credentials"
- **Solution**: Regenerate your PyPI API token and update `~/.pypirc`

**Issue**: "File already exists"
- **Solution**: PyPI doesn't allow re-uploading the same version. Bump the version number.

**Issue**: Trusted Publishing fails
- **Solution**: Verify the GitHub repository, workflow name, and environment name match PyPI settings exactly

## Additional Resources

- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [meson-python documentation](https://meson-python.readthedocs.io/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Python Packaging User Guide](https://packaging.python.org/)
