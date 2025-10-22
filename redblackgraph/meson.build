# RedBlackGraph package build configuration

# Install Python source files
python_sources = [
  '__init__.py',
  '_version.py',
  '__config__.py',
]

py.install_sources(
  python_sources,
  subdir: 'redblackgraph'
)

# Build subpackages
subdir('core')
subdir('sparse')
subdir('types')
subdir('util')
subdir('reference')
