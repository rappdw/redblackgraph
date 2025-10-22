# Next Steps - GitHub Actions Migration

## ✅ Completed
- ✅ Created `.github/workflows/` directory structure
- ✅ Implemented CI workflow with GitHub native coverage
- ✅ Created integration test workflow  
- ✅ Created release workflow
- ✅ Created badge setup documentation

## 🎯 Immediate Next Steps

### 1. Set Up Coverage Badge (Required)
**Time**: 10-15 minutes

Follow the [Badge Setup Guide](./badge_setup_guide.md) to:
1. Create a GitHub Personal Access Token with `gist` scope
2. Create a public Gist for coverage data
3. Add `GIST_SECRET` and `COVERAGE_GIST_ID` to repository secrets

**Why**: The coverage badge in the CI workflow needs these secrets to function.

### 2. Test Workflows on Feature Branch
**Time**: 1-2 hours

```bash
# Create feature branch
git checkout -b feature/github-actions-ci

# Add workflow files
git add .github/workflows/

# Commit and push
git commit -m "Add GitHub Actions CI/CD workflows"
git push -u origin feature/github-actions-ci
```

**Expected Results**:
- ✅ CI workflow should trigger automatically
- ✅ Lint job should pass
- ✅ Test jobs should pass for Python 3.10, 3.11, 3.12
- ✅ Coverage reports should be generated
- ⚠️ Coverage badge job will skip (only runs on master/main)

**To Verify**:
- Go to: https://github.com/rappdw/redblackgraph/actions
- Check that workflows run successfully
- Review job logs for any errors
- Download and inspect coverage artifacts

### 3. Review and Adjust (If Needed)
**Time**: 30 minutes - 1 hour

Based on test results:
- ✅ Check if all Python versions build successfully
- ✅ Verify coverage meets 65% threshold
- ✅ Confirm test durations are acceptable
- ✅ Review any warnings or errors in logs

**Common Adjustments**:
- If builds are slow: optimize caching
- If tests fail: check for environment-specific issues
- If coverage is incorrect: verify coverage.cfg settings

### 4. Create Pull Request
**Time**: 15 minutes

```bash
# From your feature branch
gh pr create --title "Migrate to GitHub Actions CI/CD" \
  --body "Migrates from Travis CI to GitHub Actions. See .plans/ghcicd/ for details."
```

Or manually:
1. Go to repository on GitHub
2. Click "Pull Requests" → "New Pull Request"
3. Select your feature branch
4. Add description linking to migration plan
5. Create PR (don't merge yet!)

**Purpose**: Test the PR workflow and coverage comments

### 5. Parallel Testing Period
**Time**: 2-4 weeks

During this phase:
- Keep Travis CI active
- Monitor both CI systems
- Compare results:
  - Build times
  - Test success rates  
  - Coverage percentages
  - Reliability

**Track in**: [migration_checklist.md](./migration_checklist.md) Phase 2 section

## 📋 Before Moving to Phase 3 (Cutover)

Ensure you have:
- [ ] Badge secrets configured correctly
- [ ] At least 10 successful workflow runs
- [ ] Coverage badge displaying correctly (after merging to main)
- [ ] Team comfortable with GitHub Actions
- [ ] Comparable or better performance vs Travis CI
- [ ] No increase in test flakiness
- [ ] Documentation reviewed and understood

## 🚨 Troubleshooting

### Workflow doesn't trigger
- Check that workflow file is in `.github/workflows/`
- Verify YAML syntax with: `yamllint .github/workflows/ci.yml`
- Check repository settings → Actions → General (ensure actions are enabled)

### Build fails with Meson errors
- Verify all build dependencies are installed in correct order
- Check that Python version matches matrix
- Look for system dependencies that might be missing

### Tests fail but pass locally
- Check for environment variables differences
- Verify file paths are correct (absolute vs relative)
- Look for timing-dependent tests

### Coverage badge job fails
- This is expected until you merge to master/main
- Verify secrets are set: Settings → Secrets → Actions
- Check that Gist ID is correct

### Coverage percentage differs from Travis CI
- Compare coverage.cfg settings
- Check which files are included/excluded
- Verify same test flags are used

## 💡 Optional Enhancements

Consider adding these later (Phase 4):
- [ ] Dependabot for dependency updates
- [ ] CodeQL for security scanning  
- [ ] Pre-commit hooks workflow
- [ ] Performance benchmarking
- [ ] Multi-platform testing (Windows, macOS)
- [ ] Documentation deployment

## 📞 Getting Help

If you encounter issues:
1. Check workflow logs in GitHub Actions tab
2. Review [migration_plan.md](./migration_plan.md) troubleshooting section
3. Consult [GitHub Actions documentation](https://docs.github.com/en/actions)
4. Check similar Python projects for examples

## 📝 Documentation to Update Later

After successful cutover (Phase 3):
- [ ] README.md - Replace Travis badge with GitHub Actions badge
- [ ] README.md - Replace Codecov badge with Gist badge
- [ ] CONTRIBUTING.md - Update CI instructions (if exists)
- [ ] .travis.yml - Add deprecation notice
- [ ] migration_checklist.md - Mark Phase 3 items complete

---

**Current Status**: Phase 1 - Workflows Created, Ready for Testing  
**Next Action**: Set up coverage badge secrets (15 min)  
**Blocker**: None  
**Last Updated**: 2025-10-22
