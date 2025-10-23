# GitHub Actions CI Migration - Current Status

**Date**: 2025-10-22  
**Branch**: `feature/github-actions-ci`  
**Status**: 🚧 In Progress - Build Configuration

## Summary

Migrating RedBlackGraph from Travis CI to GitHub Actions. Discovered that generated C files need to be built during CI, leading to a parallel workstream for Tempita conversion.

## Completed ✅

1. ✅ Created `.github/workflows/` directory structure
2. ✅ Implemented CI workflow (`ci.yml`)
   - Lint job (Python 3.12)
   - Test matrix (Python 3.10, 3.11, 3.12)
   - GitHub native coverage reporting
   - Coverage badge support (ready for Gist setup)
3. ✅ Created integration test workflow (`integration.yml`)
4. ✅ Created release workflow (`release.yml`)
5. ✅ Fixed deprecated actions (v3 → v4 for upload/download-artifact)
6. ✅ Created `generate_config.py` to generate `__config__.py`
7. ✅ Updated all workflows to generate `__config__.py` before build
8. ✅ Added temporary branch trigger for testing
9. ✅ Researched NumPy/SciPy Meson migration approaches

## Current Issues 🔴

### Issue #1: Generated C Files Missing in CI
**Problem**: Build fails because generated C files are gitignored:
- Core: `.c` files generated from `.c.src` templates (numpy.distutils style)
- Sparse: `.c` files generated from `.pyx` Cython files

**Current Status**: 
- Updated `redblackgraph/core/meson.build` to use `custom_target()` for template processing
- Updated `redblackgraph/sparse/csgraph/meson.build` to compile directly from `.pyx`
- Created `tools/process_template.py` (temporary solution)

**Blocker**: Need proper template processing

### Issue #2: Workflow Not Triggering
**Resolved**: Added `feature/github-actions-ci` to workflow triggers temporarily

## Decision Point: Template Processing

### Options Evaluated:
1. ❌ **Commit generated files** - Brittle, not maintainable
2. ❌ **Simple pass-through script** - Doesn't actually process templates
3. ⚠️ **Vendor numpy's conv_template.py** - Works but uses deprecated format
4. ✅ **Convert to Tempita** - Modern, NumPy 2.x aligned

### **DECISION**: Convert to Tempita (Option 4)
**Rationale**:
- Aligns with NumPy 2.x patterns
- Future-proof for NumPy 2.x C API migration
- More maintainable long-term
- Do it once, done right

### **Impact**: Separate workstream created
- Branch: `feature/tempita-conversion` (will branch off current)
- Spec: `.plans/tempita_conversion_spec.md`
- Timeline: ~3-4 weeks
- Can proceed in parallel

## Next Steps

### Path Forward - Two Parallel Tracks:

#### Track A: Complete GitHub Actions CI (This Branch)
**Can proceed independently with temporary solution**:

1. **Commit current changes** with temp solution:
   ```bash
   git add .gitignore tools/process_template.py 
   git add redblackgraph/*/meson.build
   git commit -m "WIP: Set up build-time code generation (temporary)"
   ```

2. **Test CI build** (will still fail but validates workflow)

3. **Complete remaining CI tasks**:
   - Set up Gist for coverage badge
   - Configure branch protection
   - Update README with new badges
   - Remove Travis CI configuration

4. **Merge strategy**:
   - Either: Wait for Tempita conversion then merge both
   - Or: Merge with temp solution, replace when Tempita ready

#### Track B: Tempita Conversion (New Branch)
**Independent development**:

1. **Create branch**: `feature/tempita-conversion` from `feature/github-actions-ci`
2. **Follow spec**: `.plans/tempita_conversion_spec.md`
3. **Test thoroughly**: All platforms, all Python versions
4. **Merge back**: To `feature/github-actions-ci` before final merge

### Recommended Approach:
**🎯 OPTION 1 (Recommended)**: 
- Pause CI work temporarily
- Complete Tempita conversion first (3-4 weeks)
- Resume CI with proper template processing
- Merge everything together

**Why?**: 
- Cleaner git history
- No temporary hacks in main
- Single comprehensive PR
- Proper solution from the start

**OPTION 2 (Faster but messier)**:
- Complete CI with temporary `process_template.py`
- Merge to main
- Do Tempita conversion later
- Replace temp solution in separate PR

## Files Modified (Uncommitted)

```
M  .gitignore (reverted to keep generated files ignored)
M  redblackgraph/core/meson.build (added custom_target for templates)
M  redblackgraph/sparse/csgraph/meson.build (compile from .pyx)
A  tools/process_template.py (temporary - to be replaced)
A  .plans/tempita_conversion_spec.md (new spec)
```

## Workflow Status

### CI Workflow (`ci.yml`)
- ✅ Syntax valid
- ✅ Lint job configured
- ✅ Test matrix configured  
- ✅ Coverage reporting configured
- ⚠️ **Blocked**: Build fails on missing generated C files
- 🔴 **Action**: Need template processing solution

### Integration Workflow (`integration.yml`)
- ✅ Configured
- ⚠️ Not tested (blocked by CI failure)

### Release Workflow (`release.yml`)
- ✅ Configured
- ⚠️ Not tested (blocked by CI failure)

## Coverage Badge Setup (Ready but Not Configured)

The CI workflow is ready for GitHub native coverage badges:

**Required**:
1. Create public Gist: `redblackgraph-coverage.json`
2. Create Personal Access Token with `gist` scope
3. Add secrets to repository:
   - `GIST_SECRET`: Personal Access Token
   - `GIST_ID`: Gist ID
4. Update README.md with badge URL

**See**: `.plans/ghcicd/badge_setup_guide.md`

## Testing Checklist

### CI Workflow
- [x] Workflow file syntax valid
- [x] Lint job definition correct
- [x] Test matrix set up
- [x] Coverage reporting configured
- [x] Artifact upload configured (v4)
- [ ] **BLOCKED**: Build succeeds
- [ ] Tests pass
- [ ] Coverage calculated
- [ ] Artifacts uploaded

### Build System
- [x] `generate_config.py` works
- [x] Meson build files updated
- [ ] **BLOCKED**: Template processing works
- [ ] **BLOCKED**: Cython compilation works
- [ ] **BLOCKED**: Full build succeeds

## Timeline

### Original Estimate: 1-2 days
**Actual**: Extended due to discovered dependency on template processing

### Current Status:
- **Day 1**: ✅ Workflow creation and initial setup
- **Day 2**: ⚠️ Discovered build issues, researched solutions
- **Decision**: Branch into Tempita conversion workstream

### Revised Estimates:
- **Option 1** (Recommended): 3-4 weeks (includes Tempita)
- **Option 2** (Quick): 1-2 more days (with temp solution)

## Resources

- Main spec: `.plans/ghcicd/migration_plan.md`
- Badge setup: `.plans/ghcicd/badge_setup_guide.md`
- Tempita spec: `.plans/tempita_conversion_spec.md`
- GitHub Actions: https://github.com/rappdw/redblackgraph/actions

## Notes

- Branch `feature/github-actions-ci` has temporary trigger for testing
- Remove `feature/github-actions-ci` from workflow triggers before merging
- Travis CI config preserved until migration complete
- Codecov removed (repository deactivated)
