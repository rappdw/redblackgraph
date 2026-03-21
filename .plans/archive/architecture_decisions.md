# Architecture Decision Records (ADRs) - Phase 1

**Purpose**: Document significant architectural decisions made during Phase 1 migration  
**Date Started**: 2025-10-21

---

## ADR Format

Each ADR includes:
- **Status**: Accepted, Proposed, Rejected, Superseded
- **Context**: The situation requiring a decision
- **Decision**: What was decided
- **Consequences**: Positive and negative outcomes
- **Alternatives Considered**: Other options evaluated

---

## ADR-001: Use uv for Python Version Management

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

Phase 1 requires managing multiple Python versions (3.10, 3.11, 3.12) for development and testing. Need a reliable, fast tool for installing and managing these versions.

### Decision

Use `uv` as the Python version manager instead of pyenv, conda, or asdf.

### Rationale

- **Speed**: uv is significantly faster than alternatives
- **Simplicity**: Single tool for both Python management and package installation
- **Modern**: Built in Rust, actively maintained
- **Consistency**: Works across Linux and macOS
- **Integration**: Excellent pip compatibility

### Consequences

**Positive**:
- Fast environment creation (seconds vs minutes)
- Better dependency resolution
- Simpler workflow (one tool instead of multiple)
- Modern tooling aligned with Python packaging future

**Negative**:
- Team must learn new tool
- Less established than pyenv/conda
- May have edge cases or bugs

**Neutral**:
- Requires uv installation as prerequisite
- Documentation needs to include uv setup

### Alternatives Considered

1. **pyenv**
   - Pro: Widely adopted, stable
   - Con: Slower, requires additional tools for venv management
   
2. **conda**
   - Pro: Mature, good for scientific computing
   - Con: Heavy, slower, separate ecosystem
   
3. **asdf**
   - Pro: Multi-language support
   - Con: Additional layer, less Python-focused

### Implementation

- Installation: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python install: `uv python install 3.10 3.11 3.12`
- Venv creation: `uv venv .venv-3.10 --python 3.10`
- Package install: `uv pip install --python .venv-3.10 <package>`

### Status: Implemented in Sprint 1

---

## ADR-002: No Performance Benchmarking in Sprint 1

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

Original Sprint 1 plan included Task 1.4 for performance baseline capture (6 hours effort). Need to decide if performance benchmarking is necessary before starting implementation work.

### Decision

Skip performance baseline capture in Sprint 1 and defer to future sprints or post-Phase 1.

### Rationale

- **Scope Reduction**: Sprint 1 already has sufficient work
- **Build Issues**: Cannot build on target platforms yet (blocked)
- **Research Project**: Not production-critical, performance less important
- **Faster Completion**: Sprint 1 can complete in 2 days instead of 3
- **Future Assessment**: Can measure performance if users report issues

### Consequences

**Positive**:
- Faster Sprint 1 completion
- Less upfront work
- Can focus on core migration tasks

**Negative**:
- No baseline for comparison
- May discover performance issues later
- Cannot quantify performance changes

**Neutral**:
- Performance benchmarking can be added later if needed
- Not a blocking issue for Phase 1 completion

### Alternatives Considered

1. **Benchmark Now**
   - Pro: Have baseline data
   - Con: Cannot build yet, adds time

2. **Benchmark in Sprint 4**
   - Pro: Can compare before/after
   - Con: Requires baseline from old Python versions (harder later)

3. **Benchmark Post-Phase 1**
   - Pro: Focus on getting migration done
   - Con: No data until after completion

### Implementation

- Task 1.4 removed from Sprint 1
- Performance testing deferred
- Can revisit if performance concerns arise

### Status: Implemented in Sprint 1

---

## ADR-003: Apple Silicon (ARM64) Only for Development

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

Development can target x86_64, ARM64, or both architectures. Need to decide which architecture(s) to support in development environment.

### Decision

Support only ARM64 (Apple Silicon / Linux aarch64) architecture for development. CI/CD (Travis) will continue testing x86_64.

### Rationale

- **Dev Hardware**: Development machine is ARM64
- **Simplification**: Single architecture simplifies setup
- **Coverage**: Travis CI tests x86_64 separately
- **Future**: ARM64 becoming more common (Apple Silicon, AWS Graviton)

### Consequences

**Positive**:
- Simpler development setup
- Faster builds on modern ARM hardware
- Aligned with future hardware trends

**Negative**:
- Cannot test x86_64-specific issues locally
- Developers need ARM64 machines
- Some edge cases may only appear on x86_64

**Neutral**:
- CI/CD provides x86_64 coverage
- Most issues are architecture-independent

### Alternatives Considered

1. **x86_64 Only**
   - Pro: More traditional
   - Con: Doesn't match dev hardware

2. **Both Architectures**
   - Pro: Comprehensive testing
   - Con: More complex setup, slower

3. **Docker Emulation**
   - Pro: Can test both
   - Con: Slow, complex

### Implementation

- All development environments use ARM64 Python builds
- uv installs `cpython-*-linux-aarch64-gnu` versions
- Travis CI handles x86_64 testing
- Documentation specifies ARM64 requirement

### Status: Implemented in Sprint 1

---

## ADR-004: Keep numpy.distutils for Phase 1

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

numpy.distutils is deprecated since NumPy 1.23.0 and will be removed in NumPy 2.0. Need to decide whether to migrate away from it in Phase 1 or defer to Phase 2.

### Decision

Keep using numpy.distutils in Phase 1 by:
- Pinning NumPy < 2.0 (using 1.26.x)
- Using setuptools < 60.0 (which includes distutils shim)
- Deferring Meson migration to Phase 2

### Rationale

- **Scope Management**: Meson migration is large, separate project
- **Risk Reduction**: One major change at a time
- **Timeline**: Phase 1 can complete in 2-3 weeks with this approach
- **Feasibility**: setuptools<60.0 still provides distutils compatibility
- **Testing**: Can validate Python 3.10+ support before build system migration

### Consequences

**Positive**:
- Phase 1 remains focused and achievable
- Lower risk of multiple simultaneous breaking changes
- Can complete Python version migration faster
- Deprecation warnings acceptable temporarily

**Negative**:
- Must accept numpy.distutils deprecation warnings
- Phase 2 still required for long-term solution
- Using older setuptools version
- Two-phase migration takes longer overall

**Neutral**:
- Phase 2 can learn from Phase 1 experience
- Meson migration can be better planned

### Alternatives Considered

1. **Migrate to Meson in Phase 1**
   - Pro: Single migration, modern build system
   - Con: Too much scope, high risk, longer timeline

2. **Switch to setuptools build_ext**
   - Pro: Simpler than Meson
   - Con: Cannot handle .c.src templates, doesn't work for complex C extensions

3. **Use scikit-build**
   - Pro: Intermediate complexity
   - Con: Additional dependency, still requires CMake/Meson

### Implementation

**Sprint 2**:
```python
setup_requires=[
    'numpy>=1.26.0,<2.0',
    'setuptools<60.0',  # Provides distutils shim
    'cython>=3.0'
]
```

**Phase 2** (future):
- Migrate to Meson build system
- Write meson.build files
- Handle .c.src template processing
- Remove setuptools<60.0 constraint

### Status: Implemented in Sprint 2

---

## ADR-005: Target NumPy 1.26.x (Not 2.0)

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

NumPy 2.0 is available with major changes. Need to decide whether Phase 1 targets NumPy 1.26.x or 2.0.

### Decision

Target NumPy 1.26.x and explicitly exclude NumPy 2.0 in Phase 1.

Constraint: `numpy>=1.26.0,<2.0`

### Rationale

- **Risk Management**: NumPy 2.0 has significant breaking changes
- **One Step at a Time**: Python version migration is enough for Phase 1
- **numpy.distutils**: Required for Phase 1 build, removed in NumPy 2.0
- **Ecosystem**: Many libraries not yet NumPy 2.0 compatible
- **Testing**: Can validate 1.26.x thoroughly before considering 2.0

### Consequences

**Positive**:
- Lower risk Phase 1 migration
- Smaller scope, faster completion
- Known compatibility (NumPy 1.26 works with Python 3.10+)
- Time to evaluate NumPy 2.0 impact

**Negative**:
- Will need another migration for NumPy 2.0 eventually
- Users cannot use latest NumPy with redblackgraph
- May miss NumPy 2.0 performance improvements

**Neutral**:
- NumPy 2.0 migration could be Phase 3 or separate project
- NumPy 1.26 will remain available for years

### Alternatives Considered

1. **Target NumPy 2.0 Immediately**
   - Pro: Future-proof, modern
   - Con: Too many breaking changes, requires Meson

2. **Support Both 1.26 and 2.0**
   - Pro: Maximum compatibility
   - Con: Extremely complex, testing matrix explosion

3. **No Upper Bound**
   - Pro: Maximum flexibility
   - Con: Will break when users install NumPy 2.0

### Implementation

```python
install_requires=[
    'numpy>=1.26.0,<2.0',  # Explicit upper bound
    'scipy>=1.11.0',
    # ...
]
```

### Future Work

After Phase 1, evaluate:
- NumPy 2.0 compatibility requirements
- Ecosystem readiness
- Breaking changes impact
- Whether to support NumPy 2.0

### Status: Implemented in Sprint 2

---

## ADR-006: Cython 3.x (Not 3.5 Specifically)

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

Implementation plan specified Cython 3.5, but Cython 3.5 does not exist yet (latest is 3.1.5). Need to decide which Cython version to use.

### Decision

Use Cython >= 3.0 (latest 3.x), currently 3.1.5, instead of waiting for 3.5.

### Rationale

- **Availability**: Cython 3.5 doesn't exist
- **Compatibility**: Cython 3.1.5 works with Python 3.10-3.12
- **No Blocker**: No known reason to wait for 3.5
- **Flexibility**: `>=3.0` allows future 3.x versions

### Consequences

**Positive**:
- Can proceed immediately
- Latest stable Cython 3.x
- Room for future Cython updates

**Negative**:
- Minor version differences from original plan (negligible)

**Neutral**:
- Will auto-upgrade to 3.5 when available (with `>=3.0`)

### Implementation

```python
setup_requires=[
    'cython>=3.0',  # Not specifically 3.5
]
```

### Status: Implemented in Sprint 1

---

## ADR-007: Direct Commits to Migration Branch (No PRs)

**Status**: ✅ Accepted  
**Date**: 2025-10-21  
**Sprint**: 1

### Context

Need to decide on code review process for Phase 1 work.

### Decision

Commit directly to `migration` branch without pull requests or code review.

### Rationale

- **Single Developer**: Project owner is sole developer
- **No Review Needed**: Per architectural clarification, no code review process
- **Faster Iteration**: No PR overhead
- **Git Safety**: Can still revert if needed

### Consequences

**Positive**:
- Fast development cycle
- No approval bottlenecks
- Simpler workflow

**Negative**:
- No second pair of eyes
- Easier to miss issues

**Neutral**:
- Testing provides quality gate instead of code review
- Can add reviews later if project grows

### Implementation

- Commit directly to migration branch
- Use descriptive commit messages
- Run tests before each push
- Tag sprint completions

### Status: Implemented throughout Phase 1

---

## Template for Future ADRs

```markdown
## ADR-XXX: Title

**Status**: Proposed/Accepted/Rejected/Superseded  
**Date**: YYYY-MM-DD  
**Sprint**: X

### Context

[Describe the situation and problem]

### Decision

[State the decision clearly]

### Rationale

[Why this decision was made]

### Consequences

**Positive**:
- [Benefits]

**Negative**:
- [Drawbacks]

**Neutral**:
- [Neither good nor bad]

### Alternatives Considered

1. **Alternative 1**
   - Pro: [Advantages]
   - Con: [Disadvantages]

### Implementation

[How the decision is implemented]

### Status: [Current implementation status]
```

---

**Document Status**: Active  
**Last Updated**: 2025-10-21  
**Owner**: Principal Architect / Tech Lead
