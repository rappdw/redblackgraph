# NumPy 2.x Upgrade Project

This directory contains the complete planning documentation for upgrading redblackgraph to support NumPy 2.x.

## Documents

### 📊 [analysis.md](./analysis.md)
Complete technical analysis of the NumPy 2.x upgrade requirements, including:
- Current compatibility status
- Blocking issues and their solutions
- Risk assessment
- Dependencies analysis
- Testing strategy

**Key Findings:**
- 2 blocking issues: `np.matrix` inheritance and `numpy.core.*` imports
- 1 warning: Deprecated C API function usage
- Estimated effort: 8-10 hours total

### 📋 [sprint_plan.md](./sprint_plan.md)
Discrete sprint-based implementation plan with 5 focused sprints:

1. **Sprint 1**: Remove matrix class (2 hours)
2. **Sprint 2**: Remove __config__.py (1 hour)
3. **Sprint 3**: Fix C API deprecations (2 hours)
4. **Sprint 4**: Update dependencies & CI (2 hours)
5. **Sprint 5**: Testing & validation (3 hours)

Each sprint includes detailed tasks, exit criteria, and validation steps.

## Quick Start

To execute the NumPy 2.x upgrade:

1. **Review** [analysis.md](./analysis.md) to understand the technical challenges
2. **Follow** [sprint_plan.md](./sprint_plan.md) sequentially
3. **Complete** each sprint's exit criteria before moving to the next
4. **Test** thoroughly in Sprint 5 before release

## Strategy

- **Direct removal**: No deprecation cycle, going straight to removal
- **Backwards compatible**: Still supports NumPy 1.26+
- **Breaking changes**: `matrix` class removed (use `array` instead)
- **Target release**: v0.6.0

## Timeline

- **Development**: 2-3 days
- **Testing**: 1 week
- **Release**: Week 2

## Key Decisions

1. ✅ Remove `matrix` class entirely (minimal user impact)
2. ✅ Remove `__config__.py` generation (internal only)
3. ✅ Modernize C API (future-proof)
4. ✅ Support NumPy 1.26+ through 2.x (wide compatibility)

## Success Criteria

- [ ] All 117 tests pass on NumPy 1.26.x
- [ ] All 117 tests pass on NumPy 2.0.x
- [ ] All 117 tests pass on NumPy 2.1.x
- [ ] Works on Python 3.10, 3.11, 3.12
- [ ] Works on Linux, Windows, macOS
- [ ] Clear migration guide for users

## Resources

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [NumPy 2.0 Release Notes](https://numpy.org/doc/stable/release/2.0.0-notes.html)

---

**Status**: Ready for implementation  
**Owner**: TBD  
**Last Updated**: 2025-10-24
