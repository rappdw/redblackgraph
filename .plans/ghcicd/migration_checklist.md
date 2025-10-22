# GitHub CI/CD Migration Checklist

This checklist tracks the progress of migrating from Travis CI and Codecov to GitHub Actions.

## Pre-Migration

- [ ] Review migration plan with team/stakeholders
- [ ] Choose coverage reporting solution (Codecov vs GitHub native vs Coveralls)
- [ ] Document current Travis CI baseline metrics:
  - [ ] Current build times per Python version
  - [ ] Current test success rate
  - [ ] Current coverage percentage
- [ ] Create feature branch for GitHub Actions work: `feature/github-actions-migration`

## Phase 1: Setup GitHub Actions

### Directory Structure
- [ ] Create `.github/` directory
- [ ] Create `.github/workflows/` directory

### CI Workflow
- [ ] Create `.github/workflows/ci.yml`
- [ ] Configure Python version matrix (3.10, 3.11, 3.12)
- [ ] Set up Meson build environment
- [ ] Configure pytest execution
- [ ] Configure coverage collection
- [ ] Set up coverage reporting (Codecov or alternative)
- [ ] Add pip caching for faster builds
- [ ] Configure linting job (pylint)
- [ ] Test workflow on feature branch

### Integration Test Workflow (Optional)
- [ ] Create `.github/workflows/integration.yml`
- [ ] Configure scheduled runs (e.g., weekly)
- [ ] Configure manual trigger
- [ ] Test workflow execution

### Release Workflow (Optional for Phase 1)
- [ ] Create `.github/workflows/release.yml`
- [ ] Configure wheel building
- [ ] Configure PyPI publishing (create secret)
- [ ] Test on a non-production tag

### Initial Validation
- [ ] Push feature branch to GitHub
- [ ] Verify all workflows trigger correctly
- [ ] Verify all jobs pass
- [ ] Compare test coverage with Travis CI
- [ ] Review build times vs Travis CI
- [ ] Check for any test flakiness

## Phase 2: Parallel Operation

### Monitoring Setup
- [ ] Create monitoring document/spreadsheet
- [ ] Track build success rates for both systems
- [ ] Track coverage differences
- [ ] Track build times
- [ ] Note any test flakiness issues

### Documentation
- [ ] Add GitHub Actions badges to README (alongside Travis CI)
- [ ] Document new workflow in contributor guide
- [ ] Create troubleshooting guide for common issues

### Optimization
- [ ] Fine-tune caching strategy if needed
- [ ] Optimize matrix strategy based on results
- [ ] Address any reliability issues

### Parallel Running Period
- [ ] Week 1: Monitor and collect data
- [ ] Week 2: Address any issues found
- [ ] Week 3-4: Continued monitoring, build confidence
- [ ] Final validation: Compare 10-20 commits across both systems

### Decision Point
- [ ] Review collected metrics
- [ ] Confirm GitHub Actions meets all requirements
- [ ] Get team approval to proceed with cutover
- [ ] OR: Identify and resolve blocking issues before cutover

## Phase 3: Cutover

### Badge Updates
- [ ] Update README.md to use GitHub Actions badge as primary
- [ ] Update README.md coverage badge (if using alternative to Codecov)
- [ ] Remove or mark Travis CI badge as deprecated

### Branch Protection
- [ ] Configure GitHub branch protection rules for main/master
- [ ] Require GitHub Actions CI to pass before merge
- [ ] Remove Travis CI from required checks (if configured)

### Travis CI Deprecation
- [ ] Add deprecation comment to `.travis.yml`
- [ ] Disable Travis CI builds in Travis CI settings
- [ ] Keep `.travis.yml` file for reference (don't delete yet)

### Communication
- [ ] Announce cutover to team
- [ ] Update contributor documentation
- [ ] Update any external documentation referring to CI

### Validation
- [ ] Create test PR to verify branch protection works
- [ ] Verify only GitHub Actions is required
- [ ] Test the full PR workflow

## Phase 4: Cleanup and Optimization

### Cleanup
- [ ] Archive or delete `.travis.yml` (after at least one release cycle)
- [ ] Remove codecov dependency from `pyproject.toml` (if not using Codecov)
- [ ] Clean up any Travis CI specific configurations

### Optimization
- [ ] Review and optimize caching strategies
- [ ] Consider adding additional workflow features:
  - [ ] Pre-commit hooks
  - [ ] Security scanning
  - [ ] Dependency updates (Dependabot)
  - [ ] Code quality checks
  - [ ] Performance benchmarks

### Documentation Finalization
- [ ] Update all documentation to reflect GitHub Actions
- [ ] Create runbook for common CI tasks
- [ ] Document secret management procedures
- [ ] Document release process with new workflows

### Additional Improvements
- [ ] Enable Dependabot for dependency updates
- [ ] Enable CodeQL for security scanning
- [ ] Consider multi-platform builds (macOS, Windows)
- [ ] Implement automated changelog generation
- [ ] Set up GitHub Releases automation

## Post-Migration

### Monitoring
- [ ] Monitor first 5 merges after cutover
- [ ] Track any issues or slowdowns
- [ ] Address any team feedback

### Retrospective
- [ ] Conduct migration retrospective
- [ ] Document lessons learned
- [ ] Update migration plan based on experience
- [ ] Share knowledge with other projects

### Success Validation
- [ ] All tests passing consistently
- [ ] Coverage maintained at ≥65%
- [ ] Build times acceptable
- [ ] No increase in test flakiness
- [ ] Team comfortable with new workflow
- [ ] Documentation complete and accurate

## Rollback Plan (If Needed)

If critical issues arise:

- [ ] Re-enable Travis CI in settings
- [ ] Update branch protection to require Travis CI
- [ ] Update README badges
- [ ] Investigate and resolve issues
- [ ] Plan retry of migration

## Notes and Issues

### Issues Encountered
<!-- Document any issues here for reference -->

### Decisions Made
<!-- Document key decisions and rationale -->

### Metrics Comparison

| Metric | Travis CI | GitHub Actions | Change |
|--------|-----------|----------------|--------|
| Avg Build Time (3.10) | TBD | TBD | TBD |
| Avg Build Time (3.11) | TBD | TBD | TBD |
| Avg Build Time (3.12) | TBD | TBD | TBD |
| Test Success Rate | TBD | TBD | TBD |
| Coverage % | TBD | TBD | TBD |

### Timeline

| Phase | Start Date | End Date | Status | Notes |
|-------|------------|----------|--------|-------|
| Pre-Migration | | | Not Started | |
| Phase 1: Setup | | | Not Started | |
| Phase 2: Parallel | | | Not Started | |
| Phase 3: Cutover | | | Not Started | |
| Phase 4: Cleanup | | | Not Started | |

---

**Last Updated**: 2025-10-22  
**Migration Status**: Not Started  
**Assigned To**: TBD
