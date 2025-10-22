# Csgraph subpackage - Cython extensions

# Install Python sources
python_sources = [
  '__init__.py',
  '_validation.py',
  'cycleerror.py',
  'transitive_closure.py',
]

py.install_sources(
  python_sources,
  subdir: 'redblackgraph/sparse/csgraph'
)

# Cython extensions - generate .c from .pyx during build
cython_extensions = [
  '_shortest_path',
  '_rbg_math',
  '_components',
  '_permutation',
  '_ordering',
  '_relational_composition',
  '_tools',
]

foreach ext_name : cython_extensions
  py.extension_module(
    ext_name,
    ext_name + '.pyx',  # Use .pyx source, Cython will generate .c
    dependencies: [py_dep],
    c_args: [
      '-I' + incdir_numpy,
    ],
    install: true,
    subdir: 'redblackgraph/sparse/csgraph',
  )
endforeach
