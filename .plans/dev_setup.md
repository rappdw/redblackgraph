# Development Environment Setup

**Document Version**: 1.0  
**Date**: 2025-10-21  
**Phase**: Phase 1, Sprint 1

---

## Prerequisites

- **Python Version Manager**: uv (https://astral.sh/uv/)
- **C Compiler**: gcc 11+ (tested with gcc 13)
- **Hardware**: 8GB RAM, 4 cores, 100GB disk space
- **Architecture**: ARM64 (Apple Silicon or Linux aarch64)
- **Platform**: macOS or Linux

---

## Installation Steps

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell or restart terminal
source ~/.bashrc  # Linux
# or
source ~/.zshrc   # macOS with zsh
```

Verify installation:
```bash
uv --version
# Expected: uv 0.9.x or later
```

### 2. Clone Repository

```bash
git clone https://github.com/rappdw/redblackgraph.git
cd redblackgraph
```

### 3. Install Python Versions

Install Python 3.10, 3.11, and 3.12 for Phase 1 development:

```bash
uv python install 3.10 3.11 3.12
```

Verify installations:
```bash
uv python list
```

Expected output:
```
cpython-3.10.19-linux-aarch64-gnu (python3.10)
cpython-3.11.14-linux-aarch64-gnu (python3.11)
cpython-3.12.12-linux-aarch64-gnu (python3.12)
```

### 4. Create Virtual Environments

Create separate virtual environments for each Python version:

```bash
# Python 3.10
uv venv .venv-3.10 --python 3.10

# Python 3.11
uv venv .venv-3.11 --python 3.11

# Python 3.12
uv venv .venv-3.12 --python 3.12
```

Verify environments:
```bash
ls -la | grep .venv
```

Expected:
```
.venv-3.10/
.venv-3.11/
.venv-3.12/
```

### 5. Install Dependencies

For each environment, install required dependencies using `uv pip`:

#### Python 3.10
```bash
# Install build tools
uv pip install --python .venv-3.10 --upgrade pip setuptools wheel

# Install Cython (latest 3.x)
uv pip install --python .venv-3.10 "cython>=3.0"

# Install project in development mode (AFTER Sprint 2 dependency updates)
uv pip install --python .venv-3.10 -e ".[dev,test]"
```

#### Python 3.11
```bash
uv pip install --python .venv-3.11 --upgrade pip setuptools wheel
uv pip install --python .venv-3.11 "cython>=3.0"
uv pip install --python .venv-3.11 -e ".[dev,test]"
```

#### Python 3.12
```bash
uv pip install --python .venv-3.12 --upgrade pip setuptools wheel
uv pip install --python .venv-3.12 "cython>=3.0"
uv pip install --python .venv-3.12 -e ".[dev,test]"
```

### 6. Verify Installation

Test each environment:

```bash
# Python 3.10
.venv-3.10/bin/python -c "import redblackgraph; print(f'Version: {redblackgraph.__version__}')"
.venv-3.10/bin/python -c "import redblackgraph.core; import redblackgraph.sparse; print('Extensions loaded')"

# Python 3.11
.venv-3.11/bin/python -c "import redblackgraph; print(f'Version: {redblackgraph.__version__}')"

# Python 3.12
.venv-3.12/bin/python -c "import redblackgraph; print(f'Version: {redblackgraph.__version__}')"
```

---

## Environment Activation

To activate a specific environment:

```bash
source .venv-3.10/bin/activate
```

To deactivate:
```bash
deactivate
```

**Note**: With `uv`, you can also run commands directly without activation:
```bash
# Run command in specific environment
.venv-3.10/bin/python -m pytest tests/

# Or use uv run
uv run --python .venv-3.10 python -m pytest tests/
```

---

## Known Issues (Sprint 1)

### Issue 1: Current Codebase Compatibility

**Problem**: The current `master` branch uses `numpy.distutils` which is deprecated since NumPy 1.23.0 and removed in NumPy 2.0. The current codebase cannot build with NumPy >=1.23.

**Impact**: Cannot build project with modern Python 3.10+ and NumPy 1.26+ until Sprint 2 dependency updates are complete.

**Error Message**:
```
DeprecationWarning: `numpy.distutils` is deprecated since NumPy 1.23.0
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

**Resolution**: Sprint 2 will update dependencies and build system.

### Issue 2: NPY_NO_EXPORT Macro

**Problem**: C extensions use `NPY_NO_EXPORT` macro which may not exist in NumPy >= 2.0.

**Impact**: Compilation errors when building C extensions.

**Resolution**: Sprint 3 will update C extension code for NumPy 1.26 compatibility.

### Issue 3: Cython 3.5

**Problem**: Implementation plan specified Cython 3.5, but latest available is Cython 3.1.5.

**Impact**: None - Cython 3.1.5 is compatible with Python 3.10-3.12.

**Resolution**: Use Cython 3.1.x (latest 3.x) instead of waiting for 3.5.

---

## Troubleshooting

### uv not found

**Symptom**: `bash: uv: command not found`

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

### Build fails with "gcc not found"

**Symptom**: `error: gcc: command not found`

**Solution**:
```bash
# macOS
brew install gcc@11

# Ubuntu/Debian
sudo apt install build-essential gcc-11

# Verify
gcc --version
```

### Cython compilation fails

**Symptom**: `error: cython: command not found` or Cython errors during build

**Solution**:
```bash
# Ensure Cython is installed in the environment
uv pip list --python .venv-3.10 | grep -i cython

# Reinstall if needed
uv pip install --python .venv-3.10 "cython>=3.0" --force-reinstall
```

### Import errors after installation

**Symptom**: `ModuleNotFoundError: No module named 'redblackgraph'`

**Solution**:
```bash
# Verify package is installed
uv pip show --python .venv-3.10 RedBlackGraph

# Reinstall in development mode
uv pip install --python .venv-3.10 -e ".[dev,test]" --force-reinstall
```

### Virtual environment activation issues

**Symptom**: After activation, wrong Python version

**Solution**:
```bash
# Don't rely on activation - use explicit paths
deactivate  # if activated

# Use full path to Python
.venv-3.10/bin/python --version
.venv-3.10/bin/python your_script.py
```

---

## IDE Configuration

### VS Code

Add Python interpreters:

1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Type "Python: Select Interpreter"
3. Click "Enter interpreter path..."
4. Add:
   - `/home/rappdw/dev/redblackgraph/.venv-3.10/bin/python`
   - `/home/rappdw/dev/redblackgraph/.venv-3.11/bin/python`
   - `/home/rappdw/dev/redblackgraph/.venv-3.12/bin/python`

### PyCharm

1. Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing environment"
4. Browse to `.venv-3.10/bin/python`
5. Repeat for other versions

---

## Running Tests

```bash
# Using environment's Python directly
.venv-3.10/bin/python -m pytest tests/ -v

# Or activate first
source .venv-3.10/bin/activate
bin/test -u
deactivate
```

---

## Development Workflow

1. **Make changes** to code
2. **Test in all environments**:
   ```bash
   for v in 3.10 3.11 3.12; do
     echo "Testing Python ${v}..."
     .venv-${v}/bin/python -m pytest tests/
   done
   ```
3. **Commit** changes
4. **Push** to migration branch

---

## Environment Variables

No special environment variables required for basic development.

---

## Notes

- Virtual environments are stored in project directory (`.venv-*`)
- Each environment is independent and isolated
- Cython 3.1.x is used (not 3.5 - doesn't exist yet)
- ARM64/aarch64 architecture only
- CI/CD (Travis) will test x86_64 separately

---

## Sprint 1 Status

- [x] uv installed and verified
- [x] Python 3.10, 3.11, 3.12 installed
- [x] Virtual environments created
- [ ] Dependencies installation (blocked until Sprint 2)
- [ ] Full build verification (blocked until Sprint 2)

**Next Step**: Complete Sprint 2 dependency updates, then verify builds work.

---

**Document Status**: Complete  
**Last Updated**: 2025-10-21  
**Owner**: DevOps/Lead Engineer
