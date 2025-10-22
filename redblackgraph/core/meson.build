# Core subpackage - C extensions and Python modules

# Install Python sources
python_sources = [
  '__init__.py',
  'redblack.py',
  'avos.py',
]

py.install_sources(
  python_sources,
  subdir: 'redblackgraph/core'
)

# C extension: _redblackgraph
# This extension provides optimized AVOS linear algebra operations
src_dir = 'src/redblackgraph'

_redblackgraph_sources = [
  src_dir / 'redblackgraphmodule.c',
  src_dir / 'rbg_math.c',           # Generated from .c.src
  src_dir / 'redblack.c',           # Generated from .c.src  
  src_dir / 'relational_composition.c',  # Generated from .c.src
  src_dir / 'warshall.c',           # Generated from .c.src
]

# Include directories
_redblackgraph_inc = include_directories(src_dir)

# Build the extension
py.extension_module(
  '_redblackgraph',
  _redblackgraph_sources,
  include_directories: [_redblackgraph_inc],
  dependencies: [py_dep],
  c_args: [
    '-I' + incdir_numpy,  # NumPy C API headers
  ],
  install: true,
  subdir: 'redblackgraph/core',
)
