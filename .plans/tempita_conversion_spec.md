# Tempita Conversion Specification

**Created**: 2025-10-22  
**Branch**: `feature/tempita-conversion` (off `feature/github-actions-ci`)  
**Status**: Planning Phase

## Executive Summary

Convert RedBlackGraph's C template files from NumPy distutils `.c.src` format to modern Tempita `.c.in` format to align with NumPy 2.x build patterns and prepare for NumPy 2.x C API migration.

## Background

### Current State
- RedBlackGraph uses NumPy distutils-style `.c.src` templating (`/**begin repeat` syntax)
- 4 template files totaling 3,469 lines of template code
- `redblack.c.src` (2,904 lines) is based on NumPy's einsum implementation
- Templates currently rely on pre-generated `.c` files (2.9MB+ committed locally)
- Build system migrated to Meson but templates still use legacy format

### Why This Matters
1. **NumPy 2.x Compatibility**: `numpy.distutils` removed, including `conv_template.py`
2. **Build Modernization**: Aligns with current NumPy/SciPy Meson patterns
3. **Maintainability**: Tempita syntax cleaner, easier to understand
4. **CI/CD**: Proper build-time generation discovered during GitHub Actions migration

## Research Findings: NumPy Einsum Evolution

### Current NumPy Status (2024-2025)
- **Einsum still exists** and is actively maintained in NumPy 2.x
- Location: `numpy/_core/src/multiarray/` (was `numpy/core/src/multiarray/`)
- **Still uses templating** but considering C++ template migration (Issue #29528)
- NumPy 2.0+ uses **Meson build system** exclusively
- Tempita is the standard for new template code

### Alternative Approaches Considered

#### Option A: Keep Einsum-Based Approach ✅ (Recommended)
**Pros**:
- Proven implementation (battle-tested in NumPy)
- Already working with AVOS modifications
- Well-understood performance characteristics
- Extensive SSE/AVX optimizations available
- Direct path to NumPy 2.x compatibility

**Cons**:
- Large codebase to maintain (2,904 lines)
- Needs periodic sync with NumPy upstream changes
- Template complexity

#### Option B: Custom C++ Template Implementation
**Pros**:
- Modern C++17 templates
- Compiler-optimized
- No build-time generation needed
- Smaller codebase

**Cons**:
- Complete rewrite required (~1-2 months dev time)
- Loss of SSE/AVX optimizations (would need re-implementation)
- Risk of performance regression
- Significant testing burden

#### Option C: Use opt_einsum + Custom Backend
**Pros**:
- Maintained external library
- Optimization algorithms

**Cons**:
- Python-level optimization only
- Still need C implementation for performance
- Additional dependency
- Not applicable to C-level implementation

### Recommendation
**Continue with einsum-based approach**, convert to Tempita. Rationale:
- Working implementation
- Proven performance
- Clear migration path to NumPy 2.x
- Can consider C++ conversion later as separate phase

## Scope of Work

### Files to Convert

| File | Lines | Repeat Blocks | Complexity | Priority |
|------|-------|---------------|------------|----------|
| `rbg_math.c.src` → `.c.in` | 82 | 1 | Low | P1 (Start here) |
| `rbg_math.h.src` → `.h.in` | ~30 | 1 | Low | P1 |
| `warshall.c.src` → `.c.in` | 149 | 4 | Medium | P2 |
| `relational_composition.c.src` → `.c.in` | 334 | 7 | Medium | P3 |
| `redblack.c.src` → `.c.in` | 2,904 | 50+ | High | P4 (Last) |

### Out of Scope
- NumPy 2.x C API migration (separate future phase)
- Performance optimization
- Algorithm changes
- AVOS operation modifications
- Sparse matrix Cython files (already handled by Cython compiler)

## Technical Approach

### Tempita Syntax Conversion

#### Old Format (numpy.distutils):
```c
/**begin repeat
 * #name = byte, ubyte, short#
 * #type = npy_byte, npy_ubyte, npy_short#
 * #utype = npy_ubyte, npy_ubyte, npy_ushort#
 */
@type@ @name@_avos_sum(@type@ a, @type@ b) {
    if (a == 0 || (@utype@)(~b) == 0) return b;
    // ...
}
/**end repeat*/
```

#### New Format (Tempita):
```c
{{py:
type_defs = [
    ('byte', 'npy_byte', 'npy_ubyte'),
    ('ubyte', 'npy_ubyte', 'npy_ubyte'),
    ('short', 'npy_short', 'npy_ushort'),
]
}}

{{for name, type, utype in type_defs}}
{{type}} {{name}}_avos_sum({{type}} a, {{type}} b) {
    if (a == 0 || ({{utype}})(~b) == 0) return b;
    // ...
}
{{endfor}}
```

### Build System Changes

#### Update `meson.build`:
```meson
# Install tempita for build
tempita_py = files('../../tools/tempita_processor.py')

# Generate C files from Tempita templates
rbg_math_c = custom_target('rbg_math_c',
  input: 'src/redblackgraph/rbg_math.c.in',
  output: 'rbg_math.c',
  command: [py, tempita_py, '@INPUT@', '-o', '@OUTPUT@']
)

rbg_math_h = custom_target('rbg_math_h',
  input: 'src/redblackgraph/rbg_math.h.in',
  output: 'rbg_math.h',
  command: [py, tempita_py, '@INPUT@', '-o', '@OUTPUT@']
)
# ... etc
```

#### Install Tempita:
Add to `pyproject.toml` build requirements:
```toml
[build-system]
requires = [
    "meson-python>=0.15.0",
    "meson>=1.2.0",
    "Cython>=3.0",
    "tempita>=0.5.2",  # Add this
]
```

### Validation Strategy

#### Phase 1: Conversion Verification
For each file:
1. Convert template syntax
2. Generate output with both old and new tools
3. Compare generated `.c` files (should be identical or functionally equivalent)
4. If different, validate with diff tool and understand changes

#### Phase 2: Build Verification
1. Build with new templates
2. Run full test suite
3. Compare test coverage
4. Benchmark performance (should be identical)

#### Phase 3: Distribution Verification
1. Build sdist
2. Build wheels for multiple platforms
3. Test installation from wheels
4. Verify imports and basic functionality

## Implementation Plan

### Phase 1: Setup & Proof of Concept (1-2 days)
**Branch**: Create `feature/tempita-conversion` off `feature/github-actions-ci`

**Tasks**:
- [x] Create this specification document
- [ ] Install and test tempita locally
- [ ] Create `tools/tempita_processor.py` wrapper script
- [ ] Convert `rbg_math.h.src` → `rbg_math.h.in` (smallest file, good POC)
- [ ] Verify generated output matches existing
- [ ] Update build system for this one file
- [ ] Test local build works

**Acceptance**:
- Single file conversion working
- Generated output identical or verified equivalent
- Build passes locally
- All tests pass

### Phase 2: Simple Files (2-3 days)
**Tasks**:
- [ ] Convert `rbg_math.c.src` → `rbg_math.c.in`
- [ ] Convert `warshall.c.src` → `warshall.c.in`
- [ ] Update meson.build for all simple files
- [ ] Verify generated outputs
- [ ] Test builds locally
- [ ] Document any conversion gotchas

**Acceptance**:
- 3 of 5 files converted
- All tests passing
- Performance benchmarks unchanged
- Documentation updated

### Phase 3: Medium Complexity (3-4 days)
**Tasks**:
- [ ] Convert `relational_composition.c.src` → `relational_composition.c.in`
- [ ] Handle nested repeat blocks
- [ ] Verify complex substitutions
- [ ] Full test suite validation

**Acceptance**:
- 4 of 5 files converted
- All repeat block types handled
- Tests passing
- Conversion patterns documented

### Phase 4: Large File (5-7 days)
**Tasks**:
- [ ] Analyze `redblack.c.src` structure
- [ ] Plan conversion strategy for 50+ repeat blocks
- [ ] Convert in logical sections
- [ ] Test incrementally
- [ ] Compare with NumPy's current einsum (reference check)
- [ ] Full validation

**Acceptance**:
- All files converted
- Generated C code functionally equivalent
- All tests passing
- Performance within 2% of baseline

### Phase 5: CI/CD Integration (2-3 days)
**Tasks**:
- [ ] Update GitHub Actions workflows
- [ ] Test build on all platforms (Linux, macOS, Windows)
- [ ] Verify wheel builds
- [ ] Update documentation
- [ ] Clean up old `.c.src` files (keep in git history)

**Acceptance**:
- CI builds passing on all platforms
- Wheels installable and tested
- Documentation complete
- Ready for PR to main

### Phase 6: Merge & Cleanup (1 day)
**Tasks**:
- [ ] Create PR from `feature/tempita-conversion` to `feature/github-actions-ci`
- [ ] Code review
- [ ] Address feedback
- [ ] Merge
- [ ] Delete obsolete tools/process_template.py
- [ ] Update project README

## Testing Strategy

### Unit Tests
- Run existing test suite (117 tests)
- All must pass without modification
- Coverage must remain >= 65%

### Integration Tests
- Build from source on Linux, macOS, Windows
- Test against NumPy 1.x and 2.x
- Verify AVOS operations produce correct results

### Performance Benchmarks
```python
# Baseline performance test
import numpy as np
from redblackgraph import avos_matrix_multiply

# Test matrices
sizes = [10, 100, 1000]
for n in sizes:
    a = np.random.randint(0, 256, (n, n), dtype=np.uint8)
    b = np.random.randint(0, 256, (n, n), dtype=np.uint8)
    
    # Time operation
    # Compare before/after conversion
```

Performance should be within 2% (allowing for measurement noise).

### Validation Checklist
- [ ] All 117 tests pass
- [ ] Coverage >= 65%
- [ ] Performance within 2%
- [ ] Builds on Linux (Ubuntu 20.04, 22.04, 24.04)
- [ ] Builds on macOS (x86_64, arm64)
- [ ] Builds on Windows (MSVC)
- [ ] Wheels installable
- [ ] Import works: `from redblackgraph import *`
- [ ] Example notebooks run without errors

## Dependencies

### Build-time
- Python >= 3.10
- tempita >= 0.5.2 (new)
- Meson >= 1.2.0
- meson-python >= 0.15.0
- Cython >= 3.0
- NumPy >= 1.23 (for build)

### Development
- pytest
- pytest-cov
- pylint

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Template conversion errors | High | Medium | Incremental conversion, diff validation, comprehensive testing |
| Performance regression | High | Low | Baseline benchmarks, performance tests in CI |
| Platform-specific build issues | Medium | Medium | Test on all platforms early, use CI matrix |
| Breaking changes for users | High | Low | Version as 0.6.0, clear changelog, migration guide |
| Tempita dependency issues | Medium | Low | Tempita is stable, widely used, minimal dependency |
| Large file (redblack.c) complexity | High | Medium | Do last, break into sections, extra review time |

## Success Metrics

### Must Have
- ✅ All tests pass
- ✅ Build works on Linux, macOS, Windows
- ✅ Performance unchanged (within 2%)
- ✅ CI/CD passing

### Should Have
- ✅ Documentation updated
- ✅ Conversion process documented for future reference
- ✅ Template syntax cleaner/more readable

### Nice to Have
- ✅ Improved error messages in templates
- ✅ Faster build times
- ✅ Better template debugging

## Future Phases (Out of Scope)

1. **NumPy 2.x C API Migration** (separate branch/spec)
   - Update to NEP 51 compliant C API
   - Handle deprecations
   - Test with NumPy 2.x exclusively

2. **C++ Template Conversion** (long-term)
   - Convert Tempita templates to C++17 templates
   - Evaluate performance impact
   - Reduce build complexity

3. **SIMD Optimization** (separate effort)
   - Add AVX2/AVX-512 support
   - ARM NEON support
   - Platform-specific optimizations

## Resources

### NumPy References
- [NumPy Tempita Usage](https://github.com/numpy/numpy/blob/main/numpy/_build_utils/tempita.py)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Meson Build Guide](https://numpy.org/devdocs/building/)

### Tempita Documentation
- [Tempita GitHub](https://github.com/gjhiggins/tempita)
- [Tempita Documentation](http://tempita.readthedocs.io/)

### Current Files
- `redblackgraph/core/src/redblackgraph/*.c.src` - Templates to convert
- `redblackgraph/core/meson.build` - Build file to update
- `.plans/phase2/` - Previous Meson migration docs

## Timeline

**Total Estimate**: 15-22 working days (~3-4 weeks)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Setup & POC | 1-2 days | None |
| Simple Files | 2-3 days | Phase 1 |
| Medium Complexity | 3-4 days | Phase 2 |
| Large File | 5-7 days | Phase 3 |
| CI/CD Integration | 2-3 days | Phase 4 |
| Merge & Cleanup | 1 day | Phase 5 |

**Target Completion**: ~4 weeks from start

## Sign-off

**Created by**: Cascade AI Assistant  
**Reviewed by**: _Pending_  
**Approved by**: _Pending_

---

## Appendix A: Tempita Quick Reference

### Basic Substitution
```python
{{variable_name}}
```

### Conditionals
```python
{{if condition}}
...
{{elif other_condition}}
...
{{else}}
...
{{endif}}
```

### Loops
```python
{{for item in items}}
...
{{endfor}}
```

### Python Code Blocks
```python
{{py:
# Any Python code
variable = compute_something()
}}
```

### Comments
```python
{{# This is a comment}}
```

## Appendix B: Conversion Examples

See separate document: `tempita_conversion_examples.md` (to be created during Phase 1)
