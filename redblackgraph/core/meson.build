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

# Template processors - use files() instead of find_program() to avoid 
# double python.exe invocation on Windows
tempita_processor = files('../../tools/tempita_processor.py')
src_template_processor = files('../../tools/process_src_template.py')

# Generate files from Tempita templates (.c.in, .h.in)
rbg_math_h = custom_target('rbg_math_h',
  input: src_dir / 'rbg_math.h.in',
  output: 'rbg_math.h',
  command: [py, tempita_processor, '@INPUT@', '-o', '@OUTPUT@']
)

rbg_math_c = custom_target('rbg_math_c',
  input: src_dir / 'rbg_math.c.in',
  output: 'rbg_math.c',
  command: [py, tempita_processor, '@INPUT@', '-o', '@OUTPUT@']
)

warshall_c = custom_target('warshall_c',
  input: src_dir / 'warshall.c.in',
  output: 'warshall.c',
  command: [py, tempita_processor, '@INPUT@', '-o', '@OUTPUT@']
)

relational_composition_c = custom_target('relational_composition_c',
  input: src_dir / 'relational_composition.c.in',
  output: 'relational_composition.c',
  command: [py, tempita_processor, '@INPUT@', '-o', '@OUTPUT@']
)

# Generate from NumPy-style .c.src template (follows NumPy 2.x approach)
redblack_c = custom_target('redblack_c',
  input: src_dir / 'redblack.c.src',
  output: 'redblack.c',
  command: [py, src_template_processor, '@INPUT@', '-o', '@OUTPUT@']
)

_redblackgraph_sources = [
  src_dir / 'redblackgraphmodule.c',
  rbg_math_c,                        # Generated from rbg_math.c.in (Tempita, Phase 2 ✅)
  redblack_c,                        # Generated from redblack.c.src (NumPy style, Phase 4 ✅)
  relational_composition_c,          # Generated from relational_composition.c.in (Tempita, Phase 3 ✅)
  warshall_c,                        # Generated from warshall.c.in (Tempita, Phase 2 ✅)
  rbg_math_h,                        # Generated header - forces generation before compilation
]

# Include directories - need both source and build dirs for generated headers
_redblackgraph_inc = [
  include_directories(src_dir),
  include_directories('.'),  # For generated rbg_math.h
]

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
