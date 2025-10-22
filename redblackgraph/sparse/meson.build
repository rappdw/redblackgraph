# Sparse subpackage - C/C++ extensions, Cython modules, and Python
# TODO: Sprint 3 - Add C++ extensions and Cython modules

# For now, just install Python sources
python_sources = [
  '__init__.py',
  'rbm.py',
]

py.install_sources(
  python_sources,
  subdir: 'redblackgraph/sparse'
)

# Subdirectory
subdir('csgraph')

# C++ and Cython extensions will be added in Sprint 3:
# - _sparsetools extension (C++)
# - Various Cython modules
