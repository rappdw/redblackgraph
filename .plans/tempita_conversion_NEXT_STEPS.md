# Tempita Conversion - Next Steps

**Branch**: `feature/tempita-conversion` ✅ Created  
**Current Phase**: Phase 1 - Setup & Proof of Concept  
**Status**: Ready to Begin

## What Just Happened

1. ✅ Reverted temporary build changes
2. ✅ Committed planning documents to `feature/github-actions-ci`
3. ✅ Created new branch `feature/tempita-conversion` 
4. ✅ Pushed branch to GitHub

## Branch Structure

```
master
  └─ feature/github-actions-ci (CI work paused)
       └─ feature/tempita-conversion (👈 YOU ARE HERE - active work)
```

**Merge Strategy**:
```
feature/tempita-conversion → feature/github-actions-ci → master
```

## Phase 1: Setup & Proof of Concept (1-2 days)

### Task 1: Install Tempita
```bash
pip install tempita
```

Test it works:
```bash
python -c "import tempita; print(tempita.__version__)"
```

### Task 2: Create Tempita Processor Wrapper

Create `tools/tempita_processor.py`:
```python
#!/usr/bin/env python3
"""
Tempita template processor for RedBlackGraph build system.
Wrapper around tempita to match NumPy's usage pattern.
"""
import argparse
import sys
from pathlib import Path

def process_template(input_file, output_file):
    """Process a .c.in or .h.in Tempita template."""
    try:
        import tempita
    except ImportError:
        print("Error: tempita not installed. Run: pip install tempita", 
              file=sys.stderr)
        return 1
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Read and process template
    with open(input_path, 'r', encoding='utf-8') as f:
        template_str = f.read()
    
    template = tempita.Template(template_str, name=str(input_path))
    result = template.substitute()
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"Generated {output_path} from {input_path}")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description='Process Tempita templates for RedBlackGraph'
    )
    parser.add_argument('input', help='Input template file (.c.in or .h.in)')
    parser.add_argument('-o', '--outfile', required=True, help='Output file')
    
    args = parser.parse_args()
    return process_template(args.input, args.outfile)

if __name__ == '__main__':
    sys.exit(main())
```

Make it executable:
```bash
chmod +x tools/tempita_processor.py
```

### Task 3: Convert First File (rbg_math.h.src → rbg_math.h.in)

This is the smallest file - perfect for POC!

**Step 3.1**: Copy the template:
```bash
cp redblackgraph/core/src/redblackgraph/rbg_math.h.src \
   redblackgraph/core/src/redblackgraph/rbg_math.h.in
```

**Step 3.2**: Convert the syntax in `rbg_math.h.in`

Find the repeat block (around line 10):
```c
/**begin repeat
 * #name = byte, ubyte, short, ushort, int, uint, long, ulong, longlong, ulonglong#
 * #type = npy_byte, npy_ubyte, npy_short, npy_ushort, npy_int, npy_uint, npy_long, npy_ulong, npy_longlong, npy_ulonglong#
 * #utype = npy_ubyte, npy_ubyte, npy_ushort, npy_ushort, npy_uint, npy_uint, npy_ulong, npy_ulong, npy_ulonglong, npy_ulonglong#
 */
@type@ @name@_avos_sum(@type@ a, @type@ b);
short @name@_MSB(@type@ x);
@utype@ @name@_avos_product(@type@ lhs, @type@ rhs);
/**end repeat*/
```

Replace with Tempita syntax:
```c
{{py:
type_defs = [
    ('byte', 'npy_byte', 'npy_ubyte'),
    ('ubyte', 'npy_ubyte', 'npy_ubyte'),
    ('short', 'npy_short', 'npy_ushort'),
    ('ushort', 'npy_ushort', 'npy_ushort'),
    ('int', 'npy_int', 'npy_uint'),
    ('uint', 'npy_uint', 'npy_uint'),
    ('long', 'npy_long', 'npy_ulong'),
    ('ulong', 'npy_ulong', 'npy_ulong'),
    ('longlong', 'npy_longlong', 'npy_ulonglong'),
    ('ulonglong', 'npy_ulonglong', 'npy_ulonglong'),
]
}}

{{for name, type, utype in type_defs}}
{{type}} {{name}}_avos_sum({{type}} a, {{type}} b);
short {{name}}_MSB({{type}} x);
{{utype}} {{name}}_avos_product({{type}} lhs, {{type}} rhs);
{{endfor}}
```

**Step 3.3**: Test the conversion
```bash
python tools/tempita_processor.py \
  redblackgraph/core/src/redblackgraph/rbg_math.h.in \
  -o /tmp/rbg_math.h
```

**Step 3.4**: Compare outputs
```bash
# Compare with existing generated file
diff redblackgraph/core/src/redblackgraph/rbg_math.h /tmp/rbg_math.h
```

Should be identical (or functionally equivalent)!

### Task 4: Update Build System

Update `redblackgraph/core/meson.build`:

```meson
# Tempita processor for template files
tempita_processor = find_program('../../tools/tempita_processor.py')

# Generate header from Tempita template
rbg_math_h = custom_target('rbg_math_h',
  input: src_dir / 'rbg_math.h.in',
  output: 'rbg_math.h',
  command: [py, tempita_processor, '@INPUT@', '-o', '@OUTPUT@']
)

# Rest of build...
# Include rbg_math_h as a dependency where needed
```

### Task 5: Test the Build

```bash
# Clean build
rm -rf build/

# Build with pip (uses Meson)
pip install -e . --no-build-isolation -v

# Run tests
pytest tests/
```

### Task 6: Commit Phase 1

If all tests pass:
```bash
git add tools/tempita_processor.py
git add redblackgraph/core/src/redblackgraph/rbg_math.h.in
git add redblackgraph/core/meson.build
git commit -m "Phase 1: POC - Convert rbg_math.h.src to Tempita

- Created tempita_processor.py wrapper script
- Converted rbg_math.h.src → rbg_math.h.in (Tempita syntax)
- Updated meson.build to use Tempita for header generation
- Verified output matches existing generated file
- All tests passing"

git push
```

## Phase 1 Acceptance Criteria

- [  ] Tempita installed and working
- [  ] `tempita_processor.py` created and tested
- [  ] `rbg_math.h.in` converted from `.src` format
- [  ] Generated output identical/equivalent to original
- [  ] Build succeeds with new template
- [  ] All 117 tests pass
- [  ] Changes committed and pushed

## What's Next?

After Phase 1 completes, proceed to:
- **Phase 2**: Convert simple files (`rbg_math.c.src`, `warshall.c.src`)
- See full spec: `.plans/tempita_conversion_spec.md`

## Quick Reference Commands

```bash
# Check current branch
git branch --show-current

# Run single test
pytest tests/core/test_einsum.py -v

# Build and test
pip install -e . --no-build-isolation && pytest

# Compare generated files
diff <old_file> <new_file>

# View template processor help
python tools/tempita_processor.py --help
```

## Getting Help

- **Full Spec**: `.plans/tempita_conversion_spec.md`
- **Tempita Docs**: http://tempita.readthedocs.io/
- **NumPy Examples**: Search for `.c.in` files in numpy repo
- **Current Status**: `.plans/ghcicd/CURRENT_STATUS.md`

## Notes

- Keep original `.c.src` files until conversion complete (reference)
- Don't delete generated `.c` files yet (need for comparison)
- Each phase should be a separate commit
- Run full test suite after each file conversion
- Document any conversion gotchas you find

---

**Ready to start? Begin with Task 1!** 🚀
