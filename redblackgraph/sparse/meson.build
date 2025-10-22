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
# TODO: Requires C++ code updates for NumPy 1.26+ compatibility
# Issues: NPY_C_CONTIGUOUS, NPY_NOTSWAPPED, NPY_WRITEABLE macros not available
# These were internal NumPy macros that changed in NumPy 1.26
# For now, using pre-built .so files from Phase 1
# Will fix in future sprint or use alternative implementation

# sparsetools_dir = 'sparsetools'
# 
# _sparsetools_sources = [
#   sparsetools_dir / 'sparsetools.cxx',
#   sparsetools_dir / 'rbm.cxx',
# ]
# 
# _sparsetools_inc = include_directories(sparsetools_dir)
# 
# py.extension_module(
#   '_sparsetools',
#   _sparsetools_sources,
#   include_directories: [_sparsetools_inc],
#   dependencies: [py_dep],
#   cpp_args: [
#     '-I' + incdir_numpy,
#     '-D__STDC_FORMAT_MACROS=1',
#     '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
#     '-fpermissive',
#     '-Wno-error',
#     '-std=c++11',
#   ],
#   install: true,
#   subdir: 'redblackgraph/sparse',
# )

# Subdirectory
subdir('csgraph')
