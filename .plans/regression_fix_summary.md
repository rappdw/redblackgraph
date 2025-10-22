# Regression Fix Summary - rb_matrix AVOS Multiplication

**Date**: 2025-10-21  
**Status**: ✅ **RESOLVED**  
**Affected Versions**: Python 3.10+, NumPy 1.26+, SciPy 1.15+

---

## Executive Summary

Successfully resolved regression in `rb_matrix` custom sparse matrix multiplication that caused all 5 `test_rb_matrix_square` tests to fail. **All 117 tests now pass on Python 3.10 and 3.11.**

---

## Root Cause Analysis

### The Bug

`rb_matrix` class only overrode `__matmul__` (the `@` operator) but not `__mul__` (the `*` operator). Since tests use `A * A`, they were calling the parent class `csr_matrix.__mul__()` which performs **standard matrix multiplication** instead of **AVOS (lexicographic) multiplication**.

### Why It Worked on Python 3.8

The bug existed on Python 3.8 too, but it wasn't being triggered because the test suite likely used the `@` operator or different code paths. The regression became visible when:
1. The test suite was run on the migration branch
2. Modern NumPy/SciPy changed some internal behavior
3. The `*` operator path became the primary test path

### Additional Issues Fixed

1. **pow(2, bit_position) → bit shift**: In both `rbm_math.h` and `rbg_math.c.src`, replaced `(U)pow(2, bit_position)` with `((U)1 << bit_position)` for reliable integer operations
   
2. **NPY_NO_EXPORT removal**: Removed `NPY_NO_EXPORT` from function declarations in `rbg_math.h.src` and `rbg_math.c.src` to fix NumPy 1.26 compatibility

3. **SciPy import**: Fixed deprecated import `from scipy.sparse.csr import csr_matrix` → `from scipy.sparse import csr_matrix`

---

## Changes Made

### 1. `/home/rappdw/dev/redblackgraph/redblackgraph/sparse/rbm.py`

**Added `__mul__` and `__rmul__` methods:**

```python
def __mul__(self, other):
    # Override * operator to use AVOS multiplication for sparse matrices
    # But allow scalar multiplication to use parent implementation
    if np.isscalar(other):
        return super().__mul__(other)
    return self._mul_sparse_matrix(other)

def __rmul__(self, other):
    # Handle scalar multiplication
    if np.isscalar(other):
        return super().__rmul__(other)
    # convert to this format for matrix multiplication
    return self.__class__(other)._mul_sparse_matrix(self)
```

**Fixed SciPy import:**

```python
from scipy.sparse import csr_matrix, get_index_dtype  # Was: from scipy.sparse.csr import csr_matrix
```

### 2. `/home/rappdw/dev/redblackgraph/redblackgraph/sparse/sparsetools/rbm_math.h`

**Fixed `avos_product` to use bit shift instead of pow:**

```cpp
// Before:
U result = ((y & ((U)pow(2, bit_position) - 1)) | (x << bit_position));

// After:
U result = ((y & (((U)1 << bit_position) - 1)) | (x << bit_position));
```

**Fixed type for matrix value:**

```cpp
// Ensure v stays as signed type T (not unsigned U) for proper AVOS logic
T v = Ax[jj];  // Was: U v = (U)Ax[jj];
```

### 3. `/home/rappdw/dev/redblackgraph/redblackgraph/core/src/redblackgraph/rbg_math.c.src`

**Same pow → bit shift fix:**

```c
// Before:
@utype@ result = ((y & ((@utype@)pow(2, bit_position) - 1)) | (x << bit_position));

// After:
@utype@ result = ((y & (((@utype@)1 << bit_position) - 1)) | (x << bit_position));
```

### 4. `/home/rappdw/dev/redblackgraph/redblackgraph/core/src/redblackgraph/rbg_math.h.src`

**Removed NPY_NO_EXPORT:**

```c
// Before:
NPY_NO_EXPORT @type@ @name@_avos_sum(@type@ a, @type@ b);

// After:
@type@ @name@_avos_sum(@type@ a, @type@ b);
```

### 5. `/home/rappdw/dev/redblackgraph/redblackgraph/core/src/redblackgraph/rbg_math.c.src`

**Removed static:**

```c
// Before:
static @type@ @name@_avos_sum(@type@ a, @type@ b)

// After:
@type@ @name@_avos_sum(@type@ a, @type@ b)
```

---

## Testing Results

### Before Fix

```
Python 3.10: 5 failed, 112 passed (95.7%)
Python 3.11: 5 failed, 112 passed (95.7%)
```

**Failing Tests:**
- `test_rb_matrix_square[matrix0]`
- `test_rb_matrix_square[matrix1]`
- `test_rb_matrix_square[matrix2]`
- `test_rb_matrix_square[matrix3]`
- `test_rb_matrix_square[matrix4]`

### After Fix

```
Python 3.10: 117 passed, 221 warnings (100%)
Python 3.11: 117 passed, 221 warnings (100%)
```

**All tests PASS! ✅**

---

## Technical Details

### AVOS (A Vertex, One Successor) Algebra

AVOS operations are **unsigned integer operations** with a special "red 1" value represented by -1 (the maximum value of the integer precision type).

**Key Properties:**
- AVOS sum: `min(a, b)` with special handling for 0 and -1
- AVOS product: Lexicographic concatenation of bit patterns
- -1 (max_uint) represents "red 1" (identity/special value)

### Why pow() Was Problematic

`pow(2, bit_position)` returns a `double`, which:
1. Has precision limitations for large integers (>53 bits)
2. May have rounding errors
3. Requires conversion back to integer
4. Behavior varies across platforms and compiler optimizations

Using bit shift `(1 << bit_position)` ensures:
1. Pure integer arithmetic
2. No precision loss
3. Deterministic behavior
4. Better performance

### Why __mul__ Was Missing

The original implementation assumed that sparse matrix multiplication would only use the `@` operator (PEP 465, added in Python 3.5). However:
1. Older code and tests use `*` for matrix multiplication
2. NumPy/SciPy maintain `*` for backward compatibility
3. The parent class `csr_matrix` implements `*` as standard multiplication

---

## Verification

### Manual Test

```python
import numpy as np
from scipy.sparse import coo_matrix
from redblackgraph import rb_matrix

test_matrix = rb_matrix(coo_matrix(([-1, 2, 3, -1, 2, 1, -1],
                                    ([0, 0, 0, 1, 1, 2, 3],
                                     [0, 1, 2, 1, 3, 2, 3]))))

result = test_matrix * test_matrix  # Now uses AVOS multiplication

# Verify diagonal elements are -1 (red 1)
assert result[0, 0] == -1
assert result[1, 1] == -1
assert result[3, 3] == -1
```

### Full Test Suite

```bash
# Python 3.10
.venv-3.10/bin/python -m pytest tests/ -v
# Result: 117 passed, 221 warnings

# Python 3.11
.venv-3.11/bin/python -m pytest tests/ -v
# Result: 117 passed, 221 warnings
```

---

## Impact Assessment

### Before Fix
- **Functionality**: rb_matrix multiplication completely broken
- **Test Pass Rate**: 95.7%
- **User Impact**: HIGH for rb_matrix users, LOW for others

### After Fix
- **Functionality**: rb_matrix multiplication working correctly ✅
- **Test Pass Rate**: 100% ✅
- **User Impact**: NONE - all features working

---

## Remaining Warnings (221)

All warnings are **non-critical**:

1. **NumPy out-of-bound integer conversion** (~200 warnings)
   - Test code issue, not production code
   - Will be addressed in future cleanup

2. **NumPy matrix subclass deprecation** (~20 warnings)
   - Known issue with `np.matrix`
   - Requires significant refactoring (future phase)

---

## Lessons Learned

1. **Operator Overloading**: When subclassing, override ALL relevant operators (`*`, `@`, `+`, etc.)

2. **Floating Point in Integer Algorithms**: Avoid `pow()` for integer bit operations - use bit shifts

3. **C API Compatibility**: Modern NumPy deprecates internal symbols like `NPY_NO_EXPORT`

4. **Test Coverage**: Ensure tests exercise all code paths (both `*` and `@` operators)

5. **Type Safety**: In C++ templates, ensure type consistency (T vs U) across function calls

---

## Files Modified

| File | Changes | Lines | Purpose |
|------|---------|-------|---------|
| `rbm.py` | Added __mul__, __rmul__, fixed imports | +10 | Operator overloading fix |
| `rbm_math.h` | pow → bit shift, type fix | ~2 | AVOS product fix |
| `rbg_math.c.src` | pow → bit shift, removed static | ~2 | AVOS product fix |
| `rbg_math.h.src` | Removed NPY_NO_EXPORT | ~3 | NumPy 1.26 compat |

**Total**: 4 files, ~17 lines changed

---

## Conclusion

The regression has been **completely resolved** through a combination of:
1. Proper operator overloading (`__mul__` and `__rmul__`)
2. Fixing floating-point usage in integer algorithms
3. NumPy 1.26 API compatibility fixes

**Phase 1 migration is now 100% successful** with all 117 tests passing on Python 3.10 and 3.11.

---

**Report Status**: Final  
**Last Updated**: 2025-10-21  
**Verified By**: Full test suite on Python 3.10.19 and 3.11.14  
**Next Steps**: Continue with Sprint 4 completion (documentation, Travis CI, tagging)
