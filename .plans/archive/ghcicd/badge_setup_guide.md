# GitHub Native Coverage Badge Setup Guide

Since you've chosen GitHub native coverage, the coverage badge requires a one-time setup using GitHub Gists.

## Why a Gist?

GitHub doesn't provide native coverage badges. We use a Gist to store the coverage data as JSON, which shields.io can then render as a badge.

## Setup Steps

### Step 1: Create a Personal Access Token (Classic)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Set the note: "RedBlackGraph Coverage Badge"
4. Select scopes:
   - ✅ `gist` (Create gists)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)

### Step 2: Create a Gist for Coverage Data

1. Go to https://gist.github.com/
2. Create a new public Gist
3. Filename: `redblackgraph-coverage.json`
4. Content (initial placeholder):
   ```json
   {
     "schemaVersion": 1,
     "label": "coverage",
     "message": "0%",
     "color": "red"
   }
   ```
5. Click "Create public gist"
6. **Copy the Gist ID** from the URL:
   - URL will be: `https://gist.github.com/rappdw/GIST_ID_HERE`
   - Example: If URL is `https://gist.github.com/rappdw/abc123def456`, the ID is `abc123def456` (f559859044b3e491a5dd6d75887c5145)

### Step 3: Add GitHub Secrets

Add two secrets to your repository:

1. Go to your repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"

**Secret 1:**
- Name: `GIST_SECRET`
- Value: [Paste the Personal Access Token from Step 1]

**Secret 2:**
- Name: `COVERAGE_GIST_ID`
- Value: [Paste the Gist ID from Step 2]

### Step 4: Update README Badge

Replace the old Codecov badge in `README.md`:

**Old:**
```markdown
[![Coverage](https://codecov.io/gh/rappdw/redblackgraph/branch/master/graph/badge.svg)](https://codecov.io/gh/rappdw/redblackgraph)
```

**New:**
```markdown
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rappdw/f559859044b3e491a5dd6d75887c5145/raw/redblackgraph-coverage.json)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
```

**Important:** Replace `YOUR_GIST_ID` with your actual Gist ID from Step 2.

### Step 5: Update CI Badge

Also update the Travis CI badge:

**Old:**
```markdown
[![TravisCI](https://api.travis-ci.org/rappdw/redblackgraph.svg?branch=master)](https://travis-ci.org/rappdw/redblackgraph)
```

**New:**
```markdown
[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
```

### Step 6: Initial Workflow Run

1. Commit and push the workflow files
2. The first run will update the Gist with actual coverage data
3. After the first successful run, the badge will show the correct coverage percentage

## Verifying the Setup

After the workflow runs:

1. **Check the Gist**: Visit your Gist URL and verify it shows updated JSON with current coverage
2. **Check the Badge**: The badge in your README should display the coverage percentage
3. **Click the Badge**: It should link to your GitHub Actions workflow runs

## Troubleshooting

### Badge shows "invalid"
- Verify the Gist ID in your secret is correct
- Check that the Gist filename matches exactly: `redblackgraph-coverage.json`
- Ensure the Gist is public, not secret

### Badge not updating
- Check GitHub Actions logs for the `coverage-badge` job
- Verify `GIST_SECRET` has the correct token with `gist` scope
- Ensure the workflow is running on the `master` or `main` branch

### Workflow fails on coverage-badge job
- The job only runs on master/main branches by design
- Check that secrets are set at the repository level, not environment level
- Verify the token hasn't expired

## Alternative: Simpler Badge (Without Gist)

If you prefer not to set up the Gist, you can use a simpler badge that shows the workflow status:

```markdown
[![Tests](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
```

This shows whether tests pass/fail but doesn't show the coverage percentage.

## Badge Configuration

The badge colors are automatically determined by coverage percentage:
- 🔴 Red: 0-64% (below threshold)
- 🟡 Yellow/Orange: 65-79%
- 🟢 Green: 80-100%

These thresholds match your coverage requirement of 65% minimum.

## Security Notes

- The Personal Access Token should have **only** the `gist` scope
- Store the token securely in GitHub Secrets (never commit it)
- The Gist must be **public** for the badge to display
- Rotate the token periodically for security

## Example Complete Badge Line

Your final README badges section should look like:

```markdown
[![CI](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rappdw/abc123def456/raw/redblackgraph-coverage.json)](https://github.com/rappdw/redblackgraph/actions/workflows/ci.yml)
[![PyPi](https://img.shields.io/pypi/v/redblackgraph.svg)](https://pypi.org/project/redblackgraph/)
```

---

**Estimated Setup Time**: 10-15 minutes  
**One-time Setup**: Yes (token and Gist creation only needed once)
