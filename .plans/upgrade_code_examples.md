# RedBlackGraph Upgrade: Code Examples

This document provides specific code examples for the upgrade process.

## Phase 1: Version Bumps

### pyproject.toml (New file to create for modern packaging)

```toml
[build-system]
# For Phase 1, continue using setuptools with numpy.distutils
requires = ["setuptools>=61.0", "wheel", "Cython>=0.29.21", "numpy>=1.26.0,<2.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "RedBlackGraph"
dynamic = ["version"]
description = "Red Black Graph - A DAG of Multiple, Interleaved Binary Trees"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "AGPLv3+"}
authors = [
    {name = "Daniel Rapp", email = "rappdw@gmail.com"}
]
maintainers = [
    {name = "Daniel Rapp", email = "rappdw@gmail.com"}
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Software Development :: Version Control :: Git",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.26.0,<2.0.0",
    "scipy>=1.11.0",
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]

[project.optional-dependencies]
dev = [
    "wheel>=0.29",
]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pylint>=2.15",
]

[project.urls]
Homepage = "https://github.com/rappdw/redblackgraph"
Repository = "https://github.com/rappdw/redblackgraph"

[project.scripts]
rbg = "redblackgraph.__main__:main"

[tool.setuptools]
zip-safe = false

[tool.setuptools.dynamic]
version = {attr = "redblackgraph._version.__version__"}

[tool.setuptools.packages.find]
include = ["redblackgraph*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=redblackgraph --cov-report=html --cov-report=term"
```

### Updated requirements.txt

```txt
numpy>=1.26.0,<2.0.0
scipy>=1.11.0
XlsxWriter
fs-crawler>=0.3.2
```

### Updated .travis.yml (or migrate to GitHub Actions)

```yaml
language: python
sudo: true
dist: jammy  # Ubuntu 22.04, has Python 3.10+
python:
  - "3.10"
  - "3.11"
  - "3.12"
install:
  - pip install codecov
  - pip install cython
  - pip install -e ".[test]"
script:
  - bin/test -u
after_success:
  - codecov
```

### GitHub Actions Workflow (Recommended alternative)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=redblackgraph --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

## Phase 2: Meson Migration

### pyproject.toml for Meson

```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "Cython>=3.0.0",
    "numpy>=1.26.0",
]
build-backend = "mesonpy"

[project]
name = "RedBlackGraph"
dynamic = ["version"]
description = "Red Black Graph - A DAG of Multiple, Interleaved Binary Trees"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "AGPLv3+"}
authors = [
    {name = "Daniel Rapp", email = "rappdw@gmail.com"}
]
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[project.scripts]
rbg = "redblackgraph.__main__:main"

[tool.meson-python.args]
setup = ['--default-library=static']
```

### Root meson.build

```meson
project(
  'redblackgraph',
  'c', 'cython',
  version: '0.5.0',  # or read from file
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

cc = meson.get_compiler('c')

# Compiler flags
c_args = []
if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
  c_args += [
    '-Wno-unused-function',
    '-Wno-conversion',
    '-Wno-misleading-indentation',
  ]
endif

subdir('redblackgraph')
```

### redblackgraph/meson.build

```meson
py.install_sources(
  [
    '__init__.py',
    '_version.py',
    '__main__.py',
  ],
  subdir: 'redblackgraph'
)

subdir('core')
subdir('sparse')
subdir('types')
subdir('reference')
subdir('util')
```

### redblackgraph/core/meson.build

```meson
# For C extensions with .c.src files, you need to preprocess them first
# Option 1: Create a generator script
conv_template = find_program('../../tools/conv_template.py')

# Generate C files from .c.src templates
rbg_math_c = custom_target(
  'rbg_math_c',
  input: 'src/redblackgraph/rbg_math.c.src',
  output: 'rbg_math.c',
  command: [conv_template, '@INPUT@', '@OUTPUT@'],
)

redblack_c = custom_target(
  'redblack_c',
  input: 'src/redblackgraph/redblack.c.src',
  output: 'redblack.c',
  command: [conv_template, '@INPUT@', '@OUTPUT@'],
)

# Similar for other .c.src files
relational_composition_c = custom_target(
  'relational_composition_c',
  input: 'src/redblackgraph/relational_composition.c.src',
  output: 'relational_composition.c',
  command: [conv_template, '@INPUT@', '@OUTPUT@'],
)

warshall_c = custom_target(
  'warshall_c',
  input: 'src/redblackgraph/warshall.c.src',
  output: 'warshall.c',
  command: [conv_template, '@INPUT@', '@OUTPUT@'],
)

# C extension module
_redblackgraph_sources = [
  'src/redblackgraph/redblackgraphmodule.c',
  rbg_math_c,
  redblack_c,
  relational_composition_c,
  warshall_c,
]

py.extension_module(
  '_redblackgraph',
  _redblackgraph_sources,
  c_args: c_args,
  include_directories: [inc_np, include_directories('src/redblackgraph')],
  dependencies: [py_dep],
  install: true,
  subdir: 'redblackgraph/core'
)

# Install Python files
py.install_sources(
  [
    '__init__.py',
    'avos.py',
    'redblack.py',
  ],
  subdir: 'redblackgraph/core'
)
```

### redblackgraph/sparse/csgraph/meson.build

```meson
# Cython extensions
cython = find_program('cython')

# List of Cython extensions
cython_extensions = [
  '_shortest_path',
  '_rbg_math',
  '_components',
  '_permutation',
  '_ordering',
  '_relational_composition',
  '_tools',
]

foreach ext : cython_extensions
  py.extension_module(
    ext,
    ext + '.pyx',
    c_args: c_args,
    include_directories: [inc_np],
    dependencies: [py_dep],
    install: true,
    subdir: 'redblackgraph/sparse/csgraph',
    override_options: ['cython_language=c'],
  )
endforeach

# Install Python files
py.install_sources(
  [
    '__init__.py',
    '_validation.py',
  ],
  subdir: 'redblackgraph/sparse/csgraph'
)
```

### redblackgraph/sparse/meson.build

```meson
# Generate sparsetools if needed
generate_sparsetools = find_program('generate_sparsetools.py')

generated_sources = custom_target(
  'generate_sparsetools',
  output: ['sparsetools.h', 'sparsetools_impl.h'],  # list expected outputs
  command: [generate_sparsetools, '--no-force'],
  install: false,
)

# Sparsetools extension
_sparsetools_sources = [
  'sparsetools/sparsetools.cxx',
  'sparsetools/rbm.cxx',
]

py.extension_module(
  '_sparsetools',
  _sparsetools_sources,
  cpp_args: ['-D__STDC_FORMAT_MACROS=1'] + c_args,
  include_directories: [inc_np, include_directories('sparsetools')],
  dependencies: [py_dep, generated_sources],
  install: true,
  subdir: 'redblackgraph/sparse',
  override_options: ['cpp_std=c++11'],
)

# Install Python files
py.install_sources(
  [
    '__init__.py',
    'rbm.py',
    'base.py',
    'bsr.py',
    'compressed.py',
    'construct.py',
  ],
  subdir: 'redblackgraph/sparse'
)

subdir('csgraph')
```

### tools/conv_template.py (Helper for .c.src conversion)

```python
#!/usr/bin/env python
"""
Convert NumPy .c.src template files to .c files.
This is a simplified version of numpy.distutils.conv_template functionality.
"""
import sys
import re

def process_template(input_file, output_file):
    """
    Process a .c.src file and write to .c file.
    This is a basic implementation - you may need to enhance it based on
    the specific templating features used in your .c.src files.
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Basic template processing
    # You'll need to implement the actual templating logic based on what
    # your .c.src files use. Common patterns:
    # - @TYPE@ replacements
    # - /**begin repeat ... end repeat**/ blocks
    # - etc.
    
    # For now, this is a placeholder that just copies the file
    # You need to implement the actual templating logic
    processed = content
    
    # Example: Handle simple @TYPE@ replacements
    # types = ['float', 'double', 'int32', 'int64']
    # for typ in types:
    #     ... process for each type ...
    
    with open(output_file, 'w') as f:
        f.write(processed)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: conv_template.py <input.c.src> <output.c>")
        sys.exit(1)
    
    process_template(sys.argv[1], sys.argv[2])
```

## Phase 3: NumPy 2.0 Updates

### pyproject.toml for NumPy 2.0

```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "Cython>=3.0.0",
    "numpy>=2.0.0",
]
build-backend = "mesonpy"

[project]
name = "RedBlackGraph"
# ... rest same as before ...
dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.13.0",  # Updated for NumPy 2.0 compatibility
    "XlsxWriter",
    "fs-crawler>=0.3.2",
]
```

### Example C Code Updates for NumPy 2.0

#### Before (NumPy 1.x C API):
```c
#include <numpy/arrayobject.h>

// Creating array
npy_intp dims[2] = {n, m};
PyObject *arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

// Accessing array data
double *data = (double *)PyArray_DATA(arr);
```

#### After (NumPy 2.0 C API):
```c
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

// Creating array (same)
npy_intp dims[2] = {n, m};
PyObject *arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

// Accessing array data (same, but some functions may have changed)
double *data = (double *)PyArray_DATA(arr);

// If using dtype API, may need updates
// Old: PyArray_DescrFromType(NPY_DOUBLE)
// New: Still works, but consider using PyArray_DTypeFromTypeNum
```

### Example Cython Updates for NumPy 2.0

#### Before (NumPy 1.x Cython):
```cython
# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as cnp

def my_function(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    cdef int i, j
    cdef cnp.float64_t val
    # ... code ...
```

#### After (NumPy 2.0 Cython):
```cython
# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as cnp

# Ensure numpy is initialized
cnp.import_array()

def my_function(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    cdef int i, j
    cdef cnp.float64_t val
    # ... code ...
    
# If using buffer syntax (preferred):
def my_function_buffer(cnp.float64_t[:, ::1] arr):
    cdef int i, j
    cdef cnp.float64_t val
    # ... code ...
```

## Testing After Migration

### Test Script

Create `scripts/test_installation.py`:

```python
#!/usr/bin/env python
"""
Test script to verify RedBlackGraph installation and basic functionality.
"""
import sys
import numpy as np
import scipy
import redblackgraph

def test_versions():
    """Check version requirements."""
    print("Version Information:")
    print(f"  Python: {sys.version}")
    print(f"  NumPy: {np.__version__}")
    print(f"  SciPy: {scipy.__version__}")
    print(f"  RedBlackGraph: {redblackgraph.__version__}")
    
    # Check minimum versions
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    assert np_version >= (1, 26), f"NumPy >= 1.26 required, got {np.__version__}"
    print("✓ Version requirements met")

def test_imports():
    """Test all major imports."""
    print("\nTesting imports:")
    
    from redblackgraph.core import redblack
    print("  ✓ redblackgraph.core.redblack")
    
    from redblackgraph.sparse import rb_matrix
    print("  ✓ redblackgraph.sparse.rb_matrix")
    
    from redblackgraph.sparse.csgraph import transitive_closure
    print("  ✓ redblackgraph.sparse.csgraph.transitive_closure")
    
    from redblackgraph.reference import components
    print("  ✓ redblackgraph.reference.components")
    
    print("✓ All imports successful")

def test_basic_functionality():
    """Test basic operations."""
    print("\nTesting basic functionality:")
    
    # Test sparse matrix creation
    from redblackgraph.sparse import rb_matrix
    from scipy.sparse import csr_matrix
    
    data = np.array([1, 2, 3, 4, 5, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    indptr = np.array([0, 2, 3, 6])
    
    csr = csr_matrix((data, indices, indptr), shape=(3, 3))
    rbm = rb_matrix(csr)
    
    print(f"  ✓ Created rb_matrix: shape={rbm.shape}")
    
    # Test matrix multiplication
    result = rbm @ rbm
    print(f"  ✓ Matrix multiplication: shape={result.shape}")
    
    print("✓ Basic functionality works")

def main():
    """Run all tests."""
    try:
        test_versions()
        test_imports()
        test_basic_functionality()
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

## Build and Test Commands

### Phase 1 (setuptools with numpy.distutils)

```bash
# Clean build
rm -rf build/ dist/ *.egg-info

# Install in development mode
pip install -e ".[test]"

# Run tests
pytest tests/ -v

# Build wheel
python setup.py bdist_wheel

# Test wheel installation
pip install dist/RedBlackGraph-*.whl
python scripts/test_installation.py
```

### Phase 2 (Meson)

```bash
# Clean build
rm -rf build/ dist/ *.egg-info

# Build with meson
pip install build
python -m build

# Install in development mode (editable install)
pip install -e . --no-build-isolation

# Run tests
pytest tests/ -v

# Test wheel installation
pip install dist/RedBlackGraph-*.whl
python scripts/test_installation.py
```

### Phase 3 (NumPy 2.0)

```bash
# Same as Phase 2, but ensure NumPy 2.0+ is installed
pip install "numpy>=2.0.0"
python -m build
pip install dist/RedBlackGraph-*.whl
python scripts/test_installation.py

# Additional: Performance benchmark
python scripts/benchmark.py  # if you create one
```

## Continuous Integration Examples

### Matrix Testing with GitHub Actions

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
        numpy-version: ["1.26", "2.0"]
        exclude:
          # Example: exclude certain combinations if needed
          - python-version: "3.10"
            numpy-version: "2.0"
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        if [ "${{ matrix.numpy-version }}" = "1.26" ]; then
          pip install "numpy>=1.26,<2.0"
        else
          pip install "numpy>=2.0"
        fi
        pip install -e ".[test]"
    
    - name: Run tests
      run: pytest tests/ -v --cov=redblackgraph
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
```

## Troubleshooting Common Issues

### Issue: Cython compilation fails

```bash
# Solution: Ensure Cython is up to date
pip install --upgrade "Cython>=3.0.0"

# Check Cython can find NumPy
python -c "import numpy; print(numpy.get_include())"
```

### Issue: Meson build fails

```bash
# Solution: Check Meson and ninja are installed
pip install --upgrade meson ninja meson-python

# Try verbose build
meson setup build --wipe
meson compile -C build -v
```

### Issue: Tests fail with NumPy 2.0

```bash
# Solution: Check for dtype issues
# NumPy 2.0 is stricter about dtypes
# Ensure all dtype specifications are explicit

# Example fix in Python code:
# Before: np.array([1, 2, 3])
# After:  np.array([1, 2, 3], dtype=np.int64)
```

### Issue: Import errors after installation

```bash
# Solution: Check installation location
python -c "import redblackgraph; print(redblackgraph.__file__)"

# Ensure no conflicting installations
pip list | grep -i redblackgraph

# Reinstall cleanly
pip uninstall redblackgraph -y
pip install -e . --no-build-isolation
```
