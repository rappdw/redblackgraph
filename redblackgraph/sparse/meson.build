# Sparse subpackage - C/C++ extensions, Cython modules, and Python

# Install Python sources
python_sources = [
  '__init__.py',
  'rbm.py',
  'avos.py',
  'generate_sparsetools.py',
]

py.install_sources(
  python_sources,
  subdir: 'redblackgraph/sparse'
)

# C++ extension: _sparsetools
# This provides optimized sparse matrix operations for AVOS algebra
sparsetools_dir = 'sparsetools'

_sparsetools_sources = [
  sparsetools_dir / 'sparsetools.cxx',
  sparsetools_dir / 'rbm.cxx',
]

_sparsetools_inc = include_directories(sparsetools_dir)

# Compiler-specific C++ flags
cpp = meson.get_compiler('cpp')
cpp_std_flag = []
if cpp.get_id() == 'msvc'
  cpp_std_flag = ['/std:c++14']  # MSVC syntax
else
  cpp_std_flag = ['-std=c++11']  # GCC/Clang syntax
endif

py.extension_module(
  '_sparsetools',
  _sparsetools_sources,
  include_directories: [_sparsetools_inc],
  dependencies: [py_dep],
  cpp_args: [
    '-I' + incdir_numpy,
    '-D__STDC_FORMAT_MACROS=1',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  ] + cpp_std_flag,
  install: true,
  subdir: 'redblackgraph/sparse',
)

# Subdirectory
subdir('csgraph')
