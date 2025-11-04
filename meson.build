project(
  'redblackgraph',
  'c', 'cpp', 'cython',
  # Get version from git tag dynamically
  version: run_command('git', 'describe', '--tags', '--abbrev=0', check: false).stdout().strip().substring(1),
  license: 'AGPL-3.0-or-later',
  meson_version: '>= 1.2.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',  # c++14 is supported by MSVC, GCC, and Clang
    'warning_level=2',
  ],
)

# Python detection
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Filesystem module for file checks
fs = import('fs')

message('Project version:', meson.project_version())
message('Python path:', py.full_path())
message('Python version:', py.language_version())

# Check Python version
py_version = py.language_version()
if py_version.version_compare('<3.10')
  error('RedBlackGraph requires Python >= 3.10')
endif

# NumPy dependency and include directory
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

message('NumPy include directory:', incdir_numpy)

# For extensions, we'll pass incdir_numpy directly to include_directories
# Don't wrap external absolute paths in include_directories()

# Compiler configuration
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Add common compile args
add_project_arguments(
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  language: ['c', 'cpp']
)

# Check for required headers
if not cc.has_header('Python.h', dependencies: py_dep)
  error('Python.h not found')
endif

# Cython - find it relative to Python installation
cython_path = run_command(py,
  ['-c', 'import sys; from pathlib import Path; print(Path(sys.executable).parent / "cython")'],
  check: true
).stdout().strip()

cython = find_program(cython_path, 'cython', required: false)
if not cython.found()
  # Fallback: use python -m cython
  message('Using python -m cython instead of cython binary')
endif

# Build the main package
subdir('redblackgraph')
