# Git Branching Strategy for Phase 1 Migration

**Document Version**: 1.0  
**Date**: 2025-10-21  
**Phase**: Phase 1, Sprint 1

---

## Overview

Phase 1 migration work occurs on a dedicated long-lived branch (`migration`) to isolate changes until all sprints are complete and validated.

---

## Branch Structure

```
master (production-ready releases)
  └── migration (Phase 1 migration work)
```

### Branch Details

**master**
- **Purpose**: Production-ready code, stable releases
- **Protection**: Manual releases only
- **Current Python Support**: 3.6, 3.7, 3.8, 3.9, 3.10
- **Build System**: numpy.distutils (deprecated)

**migration** *(previously: phase1-upgrade)*
- **Purpose**: Phase 1 migration development (Sprints 1-4)
- **Lifespan**: Duration of Phase 1 (2.5-3 weeks)
- **Target Python**: 3.10, 3.11, 3.12
- **Build System**: numpy.distutils (Phase 1), then Meson (Phase 2)
- **Status**: Currently active

---

## Workflow

### Daily Development

1. **Ensure on migration branch**:
   ```bash
   git checkout migration
   git pull origin migration
   ```

2. **Make changes and commit**:
   ```bash
   git add <files>
   git commit -m "[Sprint X.Y] Descriptive message"
   ```

3. **Push to remote**:
   ```bash
   git push origin migration
   ```

### Branch Navigation

Switch between branches:
```bash
# Go to master (for reference/baseline)
git checkout master

# Return to migration
git checkout migration
```

View current branch:
```bash
git branch
# or
git status
```

---

## Commit Message Conventions

Use descriptive commit messages with sprint context:

### Format

```
[Sprint X.Y] Brief description

Optional longer explanation if needed.

Related to: Task X.Y
```

### Examples

```
[Sprint 1.1] Set up Python 3.10-3.12 environments with uv

Created three virtual environments in project directory.
Verified builds succeed on all Python versions.

Related to: Task 1.1
```

```
[Sprint 2.1] Update setup.py with new dependency constraints

- Set numpy>=1.26.0, scipy>=1.11.0
- Remove dataclasses conditional dependency
- Update Python classifiers to 3.10, 3.11, 3.12

Related to: Task 2.1
```

```
[Sprint 3.2] Replace deprecated np.int with np.int64

Updated all occurrences of np.int/np.float to use
explicit dtypes or np.int64/np.float64.

Related to: Task 3.2
```

---

## Sprint Milestones

At the end of each sprint, tag the branch to mark completion:

### Sprint 1
```bash
git tag -a phase1-sprint1-complete -m "Sprint 1: Foundation & Baseline Complete"
git push origin phase1-sprint1-complete
```

### Sprint 2
```bash
git tag -a phase1-sprint2-complete -m "Sprint 2: Dependencies Updated"
git push origin phase1-sprint2-complete
```

### Sprint 3
```bash
git tag -a phase1-sprint3-complete -m "Sprint 3: Code Modernized"
git push origin phase1-sprint3-complete
```

### Sprint 4
```bash
git tag -a phase1-sprint4-complete -m "Sprint 4: Testing & Validation Complete"
git push origin phase1-sprint4-complete
```

---

## Merge Strategy

### During Development

- **Direct commits** to `migration` branch (no PR process)
- **Merge commits** preferred over squashing
- **No code review** required (single-person project)
- **Commit frequently** with working code

### Final Merge to Master

After Sprint 4 completion and all acceptance criteria met:

```bash
# Ensure migration branch is up-to-date
git checkout migration
git pull origin migration

# Run final tests
bin/test -u

# Switch to master
git checkout master
git pull origin master

# Merge migration branch
git merge migration

# Tag the release
git tag -a v2.0.0 -m "Phase 1: Python 3.10+ migration complete"

# Push to remote
git push origin master --tags
```

---

## Rollback Procedures

### Undo Last Commit (Before Push)

```bash
# Soft reset - keeps changes staged
git reset --soft HEAD~1

# Mixed reset - keeps changes unstaged
git reset HEAD~1

# Hard reset - discards all changes
git reset --hard HEAD~1
```

### Undo Last Commit (After Push)

```bash
# Create revert commit
git revert HEAD
git push origin migration
```

### Reset to Sprint Milestone

```bash
# List available tags
git tag -l

# Reset to specific sprint completion
git reset --hard phase1-sprint2-complete

# Force push (use with caution!)
git push origin migration --force
```

### Rollback to Specific Commit

```bash
# Find commit hash
git log --oneline

# Reset to commit
git reset --hard <commit-hash>
git push origin migration --force
```

### Abandon Phase 1 (Emergency Only)

If Phase 1 needs to be completely abandoned:

```bash
# Backup current work
git checkout master
git branch migration-backup-$(date +%Y%m%d) migration

# Delete migration branch
git branch -D migration

# Start fresh (if needed)
git checkout -b migration-v2
git push -u origin migration-v2
```

---

## Testing Before Commits

Always test before committing:

```bash
# Test in all Python versions
for v in 3.10 3.11 3.12; do
  echo "Testing Python ${v}..."
  .venv-${v}/bin/python -m pytest tests/
done

# If all pass, commit
git add .
git commit -m "[Sprint X.Y] Your message"
git push origin migration
```

---

## CI/CD Integration

### Travis CI

- **Configuration**: `.travis.yml`
- **Trigger**: Automatic on push to any branch
- **Python Versions**: Currently 3.6, 3.7, 3.8 (master) → will update to 3.10, 3.11, 3.12 (migration)
- **Test Command**: `bin/test -u`
- **Code Coverage**: Uploaded to codecov

### Branch-Specific Behavior

**master branch**:
- Tests run on Python 3.6, 3.7, 3.8
- Uses current dependencies
- Expected to pass

**migration branch**:
- Initially: May have failing builds during Sprint 2-3
- After Sprint 2: Should build successfully
- After Sprint 4: All tests passing on Python 3.10, 3.11, 3.12

---

## Important Notes

### Breaking Changes

Phase 1 is a **breaking change release** (major version bump from 1.x to 2.0):
- Drops Python 3.6, 3.7, 3.8, 3.9 support
- Adds Python 3.10, 3.11, 3.12 support
- Updates to NumPy 1.26, SciPy 1.11
- Removes dataclasses backport

### Backward Compatibility

- **No backward compatibility** maintained in API
- **Old Python versions**: Python 3.6-3.9 users should continue using v1.x
- **PyPI**: Old wheels (v1.x) remain available for legacy Python versions

### Branch Protection

No branch protection rules configured. Developer responsibilities:
1. Test locally before pushing
2. Ensure tests pass with `bin/test -u`
3. Only commit working code
4. Document breaking changes

### Coordination

- **No feature branches**: All work on `migration` directly
- **No conflicts**: Single developer, sequential work
- **No merge conflicts**: Keep up-to-date with `git pull origin migration`

---

## Questions & Support

**Project Owner**: Daniel Rapp (rappdw@gmail.com)

**Common Questions**:

**Q: Can I create feature branches off migration?**
A: Not necessary - direct commits to migration are fine.

**Q: What if I break something?**
A: Use `git reset` or `git revert` to undo. Tests should catch issues before commit.

**Q: Should I squash commits?**
A: No - keep commit history intact for traceability.

**Q: When should I merge to master?**
A: Only after Sprint 4 completion and final approval.

---

## Sprint 1 Status

- [x] `migration` branch exists
- [x] Branch is checked out and active
- [x] Remote tracking configured
- [x] Commit message conventions documented
- [x] Rollback procedures defined
- [ ] Sprint 1 completion tag (pending Sprint 1 completion)

---

**Document Status**: Complete  
**Last Updated**: 2025-10-21  
**Owner**: Lead Engineer
