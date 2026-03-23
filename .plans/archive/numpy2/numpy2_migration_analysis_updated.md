# NumPy 2.0 Migration Analysis - Updated with Official Guide

**Date**: 2025-11-02  
**Based on**: [Official NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)  
**Status**: Analysis Complete

---

## Critical Findings from Official Guide

### 1. C API Changes Are More Complex Than Initially Assessed

The original analysis underestimated the C API changes. Key issues:

#### A. `PyArray_Descr` Structure Changes
**Impact**: HIGH - Structure is now opaque
- Direct access to `descr->elsize` **no longer works** in NumPy 2.0
- **Must use**: `PyDataType_ELSIZE(descr)` accessor macro
- Our code has **5 direct elsize accesses** in `redblack.c.src`
- Returns `npy_intp` (not `int`) in NumPy 2.0

#### B. `NPY_NTYPES` Constant Removed
**Impact**: HIGH - Used in static table sizing
- Used in **5 static function pointer tables** in `redblack.c.src`
- No direct replacement mentioned in migration guide
- Likely need to use `NPY_NTYPES_LEGACY` but must verify

#### C. `numpy/noprefix.h` Removed
**Impact**: CRITICAL - Build failure in NumPy 2.0
- Header completely **removed** in NumPy 2.0
- Provided unprefixed macro versions (e.g., `NOTYPE` vs `NPY_NOTYPE`)
- Used in **4 C source files** in our codebase
- **Must remove** all includes

### 2. Compatibility Layer: `npy_2_compat.h`

NumPy provides a compatibility header for dual NumPy 1.x/2.x support:

```c
#include <numpy/npy_2_compat.h>
```

**Key Points**:
- Provides new API definitions when compiling with NumPy 1.x
- Requires `import_array()` to be properly called
- No effect when compiling with NumPy 2.x
- Can be vendored into codebase if needed
- **Must include `ndarrayobject.h`** (not just `ndarraytypes.h`)

### 3. Required Header Changes

**Before NumPy 2.0**:
```c
#include <numpy/ndarraytypes.h>
#include <numpy/noprefix.h>  // REMOVED in 2.0
```

**For NumPy 2.0 Compatibility**:
```c
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>  // Required for full API
#include <numpy/npy_2_compat.h>   // Compatibility layer
```

---

## Updated Issue Assessment

### Issue #1: `np.matrix` Inheritance
**Status**: As originally assessed  
**Complexity**: LOW  
**Plan**: Remove class (already completed in original plan)

### Issue #2: `__config__.py` Generation  
**Status**: As originally assessed  
**Complexity**: LOW  
**Plan**: Remove generation script (already completed in original plan)

### Issue #3: `numpy/noprefix.h` Removal ⚠️ **NEW CRITICAL**
**Status**: **Blocking** - Was underestimated as "Warning"  
**Complexity**: MEDIUM  
**Impact**: Build **fails** in NumPy 2.0

**Files Affected**:
1. `redblackgraph/core/src/redblackgraph/redblackgraphmodule.c`
2. `redblackgraph/core/src/redblackgraph/redblack.c.src`
3. `redblackgraph/core/src/redblackgraph/warshall.c.src`
4. `redblackgraph/core/src/redblackgraph/relational_composition.c.src`

**Also template files**:
5. `redblackgraph/core/src/redblackgraph/warshall.c.in`
6. `redblackgraph/core/src/redblackgraph/relational_composition.c.in`

**Fix**: Remove all `#include <numpy/noprefix.h>` lines

### Issue #4: `PyArray_Descr->elsize` Direct Access ⚠️ **NEW CRITICAL**
**Status**: **Blocking** - Not in original analysis  
**Complexity**: MEDIUM  
**Impact**: Compilation **error** in NumPy 2.0

**Locations** (in `redblack.c.src`):
- Line ~2289: `NpyIter_GetDescrArray(iter)[0]->elsize`
- Line ~2343: `NpyIter_GetDescrArray(iter)[0]->elsize`
- Line ~2400: `NpyIter_GetDescrArray(iter)[0]->elsize`
- Line ~2456: `NpyIter_GetDescrArray(iter)[0]->elsize`
- Line ~2855: `NpyIter_GetDescrArray(iter)[0]->elsize`

**Fix**: Replace with `PyDataType_ELSIZE(NpyIter_GetDescrArray(iter)[0])`

**Compatibility Macro** (for NumPy 1.x):
```c
#if NPY_ABI_VERSION < 0x02000000
    #define PyDataType_ELSIZE(descr) ((descr)->elsize)
#endif
```

### Issue #5: `NPY_NTYPES` Constant Removed ⚠️ **NEW CRITICAL**
**Status**: **Blocking** - Not in original analysis  
**Complexity**: HIGH  
**Impact**: Compilation **error** in NumPy 2.0

**Locations** (in `redblack.c.src`):
- Line ~1515: `_contig_outstride0_unary_specialization_table[NPY_NTYPES]`
- Line ~1546: `_binary_specialization_table[NPY_NTYPES][5]`
- Line ~1583: `_outstride0_specialized_table[NPY_NTYPES][4]`
- Line ~1619: `_allcontig_specialized_table[NPY_NTYPES][4]`
- Line ~1655: `_unspecialized_table[NPY_NTYPES][4]`
- Line ~1697: `if (type_num >= NPY_NTYPES)`

**Investigation Needed**:
- NumPy 2.0 migration guide doesn't explicitly document replacement
- Likely need to use approach from `npy_2_compat.h`
- May need to define compatibility macro mapping to `NPY_NTYPES_LEGACY`

### Issue #6: `PyArray_FROM_OF` Deprecation
**Status**: As originally assessed (deprecation warning only)  
**Complexity**: LOW  
**Plan**: Replace with `PyArray_FROM_OTF` (already in original plan)

---

## Revised Migration Strategy

### Phase 1: Python Code Changes (Sprint 1-2 from original plan)
**Effort**: 2 hours  
**Status**: ✅ Completed in original approach

1. Remove `matrix` class
2. Remove `__config__.py` generation

### Phase 2: C API Header Updates (NEW)
**Effort**: 1 hour  
**Complexity**: LOW-MEDIUM

**Task 2.1**: Remove `numpy/noprefix.h` includes
- Remove from all 6 C source files (4 `.c.src`, 2 `.c.in`)
- Add `#include <numpy/npy_2_compat.h>`
- Ensure `#include <numpy/ndarrayobject.h>` is present

**Task 2.2**: Verify `import_array()` calls
- Check all C extensions properly call `import_array()`
- Required for `npy_2_compat.h` to work

### Phase 3: C API Structure Access Updates (NEW)
**Effort**: 2-3 hours  
**Complexity**: MEDIUM

**Task 3.1**: Fix `PyArray_Descr->elsize` accesses
- Replace 5 direct accesses with `PyDataType_ELSIZE()` macro
- Add compatibility macro for NumPy 1.x:
  ```c
  #if NPY_ABI_VERSION < 0x02000000
      #define PyDataType_ELSIZE(descr) ((descr)->elsize)
  #endif
  ```

**Task 3.2**: Fix `NPY_NTYPES` usage
- **Option A**: Use `NPY_NTYPES_LEGACY` with compatibility check:
  ```c
  #if !defined(NPY_NTYPES) && defined(NPY_NTYPES_LEGACY)
      #define NPY_NTYPES NPY_NTYPES_LEGACY
  #endif
  ```
- **Option B**: Research if there's a better NumPy 2.0 approach
- **Decision needed**: Verify Option A works before proceeding

### Phase 4: C API Deprecation Fixes (from original plan)
**Effort**: 1 hour  
**Complexity**: LOW

**Task 4.1**: Replace `PyArray_FROM_OF`
- Update 2 calls to use `PyArray_FROM_OTF`
- Syntax: `PyArray_FROM_OTF(obj, NPY_NOTYPE, NPY_ARRAY_ENSUREARRAY)`

### Phase 5: Testing
**Effort**: 3-4 hours  
**Complexity**: MEDIUM

**Test Matrix**:
```
Python 3.10 + NumPy 1.26.x ✓
Python 3.10 + NumPy 2.1.x  ✓
Python 3.11 + NumPy 1.26.x ✓
Python 3.11 + NumPy 2.1.x  ✓
Python 3.12 + NumPy 1.26.x ✓
Python 3.12 + NumPy 2.1.x  ✓
```

---

## Recommended Approach

### Option A: Full Migration (RECOMMENDED for v0.6.0)
**Total Effort**: 9-11 hours  
**Risk**: Medium  
**Benefit**: Complete NumPy 2.x support

**Steps**:
1. Complete Phase 1 (Python changes) - 2 hours
2. Complete Phase 2 (Header updates) - 1 hour
3. Complete Phase 3 (Structure access) - 3 hours
4. Complete Phase 4 (Deprecations) - 1 hour
5. Complete Phase 5 (Testing) - 4 hours

### Option B: Staged Migration (SAFER)
**Total Effort**: Same, but split across releases  
**Risk**: Low  
**Benefit**: More testing time between changes

**v0.6.0** (NumPy 1.x only):
- Phase 1: Python changes
- Phase 4: C API deprecations
- Keep `numpy<2.0` constraint
- Benefit: Cleaner code, no user disruption

**v0.7.0** (NumPy 2.x support):
- Phase 2: Header updates
- Phase 3: Structure access updates
- Update to `numpy>=1.26,<3.0`
- Benefit: More time to test C API changes

---

## Key Risks & Mitigations

### Risk 1: `NPY_NTYPES_LEGACY` Approach May Not Work
**Likelihood**: Medium  
**Impact**: High (blocks migration)  
**Mitigation**:
1. Test build with NumPy 2.1 after implementing
2. If it fails, research NumPy's `npy_2_compat.h` source for correct approach
3. Fallback: Hardcode table size to known NumPy type count (not ideal)

### Risk 2: Template File Generation Issues
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**:
- `.c.src` files are templates processed by tempita
- Changes to `.c.src` and `.c.in` should auto-generate correct `.c` files
- Verify generated files after build

### Risk 3: Platform-Specific C API Issues
**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Test on all platforms (Linux, macOS, Windows)
- Use CI matrix testing before release

---

## Updated Sprint Plan

### Sprint 1: Python Code Cleanup (COMPLETED)
- Remove `matrix` class ✓
- Update tests ✓

### Sprint 2: Config Generation Removal (COMPLETED)
- Remove `generate_config.py` ✓
- Update workflows ✓

### Sprint 3: C API Headers (NEW - CRITICAL)
**Duration**: 1 hour

1. Remove `#include <numpy/noprefix.h>` from:
   - `redblackgraphmodule.c`
   - `redblack.c.src`
   - `warshall.c.src`, `warshall.c.in`
   - `relational_composition.c.src`, `relational_composition.c.in`

2. Add to each file (after other numpy includes):
   ```c
   #include <numpy/ndarrayobject.h>  // If not present
   #include <numpy/npy_2_compat.h>
   ```

3. Verify `import_array()` is called in module init

### Sprint 4: C API Structure Access (NEW - CRITICAL)
**Duration**: 2-3 hours

1. Add compatibility macro to `redblack.c.src` (after includes):
   ```c
   /* NumPy 2.0 compatibility */
   #if NPY_ABI_VERSION < 0x02000000
       #define PyDataType_ELSIZE(descr) ((descr)->elsize)
   #endif
   
   #if !defined(NPY_NTYPES) && defined(NPY_NTYPES_LEGACY)
       #define NPY_NTYPES NPY_NTYPES_LEGACY
   #endif
   ```

2. Replace 5 `->elsize` accesses with `PyDataType_ELSIZE()`

3. Test build with NumPy 2.1

### Sprint 5: C API Deprecations (ORIGINAL)
**Duration**: 1 hour

1. Replace `PyArray_FROM_OF` with `PyArray_FROM_OTF`

### Sprint 6: Testing (ORIGINAL)
**Duration**: 3-4 hours

1. Test full matrix (3 Python × 2 NumPy versions)
2. Run all 117 tests
3. Verify on all platforms

---

## Decision Required

**Question**: Should we proceed with full migration (Option A) or staged migration (Option B)?

**Recommendation**: **Option A (Full Migration)** for v0.6.0
- Work is already in progress
- Total effort is manageable (9-11 hours)
- Users get NumPy 2.x support sooner
- Avoids having two breaking change releases close together

**However**: If `NPY_NTYPES_LEGACY` approach fails in Sprint 4, we may need to:
1. Seek help from NumPy community
2. Consider alternative static table approach
3. Or fallback to Option B (ship v0.6.0 without NumPy 2.x, fix in v0.7.0)

---

## Conclusion

The NumPy 2.0 migration is **more complex than originally estimated** due to:
1. Removed `numpy/noprefix.h` header (critical)
2. Opaque `PyArray_Descr` structure (critical)
3. Removed `NPY_NTYPES` constant (critical)

However, NumPy provides a good compatibility layer (`npy_2_compat.h`) that should make dual 1.x/2.x support feasible.

**Next Steps**:
1. Review this analysis
2. Decide on Option A vs Option B
3. If Option A: Proceed with Sprint 3 (C API Headers)
4. Test build with NumPy 2.1 before going further
