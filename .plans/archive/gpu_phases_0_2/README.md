# GPU Implementation Plan - NumPy 2.x Era

**Status**: Analysis Phase - **UPDATED FOR DGX SPARK**  
**Last Updated**: November 2025  
**Context**: Post NumPy 2.x migration with refined parity-constrained AVOS algebra

## 🚨 CRITICAL UPDATE (Nov 3, 2025)

**New context that significantly improves the plan**:

1. **Hardware: DGX Spark (Grace Hopper)** with unified CPU/GPU memory
2. **Structure: Upper triangular matrices** for all genealogy DAGs
3. **Scale: Billion-row target** (1B×1B at 0.1% density)

**Impact**: Architecture simplified, performance improved, same timeline.

👉 **Read [00_UPDATED_CONTEXT.md](00_UPDATED_CONTEXT.md) FIRST** for complete impact analysis.

## Executive Summary

This directory contains the comprehensive plan for implementing GPU acceleration for redblackgraph's AVOS matrix operations. With the project now on NumPy 2.x and the mathematics tightened around asymmetric identity behavior, we're positioned to create a high-performance GPU backend that:

1. **Leverages modern NumPy 2.x features** for better interoperability
2. **Correctly implements parity-constrained AVOS operations** with asymmetric identities
3. **Integrates with SciPy's existing CuPy support** where beneficial
4. **Maintains API compatibility** with existing `rb_matrix` interface
5. **Targets realistic genealogy workloads** (sparse matrices, 100-50K vertices)

## Key Mathematical Refinements Since Original Plan

The AVOS algebra now has **asymmetric identity behavior**:

### Identity Semantics (Asymmetric)
- **LEFT identity** (`-1` or `1` on left): Starting point marker, no filtering
- **RIGHT identity** (`-1` or `1` on right): Gender/parity filter

### Parity Constraints
```
RED_ONE (-1) on RIGHT:
  - even ⊗ RED_ONE = even (identity for even/male)
  - odd ⊗ RED_ONE = 0 (annihilator for odd/female)

BLACK_ONE (1) on RIGHT:
  - odd ⊗ BLACK_ONE = odd (identity for odd/female)  
  - even ⊗ BLACK_ONE = 0 (annihilator for even/male)
```

### Special Cases
```
RED_ONE ⊗ RED_ONE = RED_ONE     (male self-loop)
BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE (female self-loop)
RED_ONE ⊗ BLACK_ONE = 0         (cross-gender undefined)
BLACK_ONE ⊗ RED_ONE = 0         (cross-gender undefined)
```

**Impact on GPU implementation**: Parity checks add conditional logic to kernels but are highly parallelizable.

## NumPy 2.x Advantages

1. **Improved Array API compatibility** - Better tensor operations
2. **New C API features** - More efficient extension development
3. **Better dtype handling** - Cleaner type management
4. **Performance improvements** - Native optimizations we can build on
5. **Modern build system** - Already using Meson

## Document Structure

- **[01_architecture.md](01_architecture.md)** - Overall architecture and design decisions
- **[02_cuda_kernels.md](02_cuda_kernels.md)** - Detailed CUDA kernel specifications
- **[03_cupy_integration.md](03_cupy_integration.md)** - CuPy wrapper and Python API
- **[04_performance_strategy.md](04_performance_strategy.md)** - Optimization targets and benchmarks
- **[05_testing_plan.md](05_testing_plan.md)** - Validation and correctness testing
- **[06_implementation_phases.md](06_implementation_phases.md)** - Step-by-step roadmap
- **[07_triangularization.md](07_triangularization.md)** - GPU-accelerated preprocessing (Phase 3a)

## Quick Reference: Current Implementation

### Implementations
1. **`redblackgraph.reference`** - Pure Python (educational)
2. **`redblackgraph.core`** - NumPy C API + einsum (dense arrays)
3. **`redblackgraph.sparse`** - CSR sparse (C++ + Cython)

### Key Files
- `redblackgraph/reference/rbg_math.py` - Reference AVOS operations
- `redblackgraph/core/src/redblackgraph/rbg_math.c.in` - Templated C implementation
- `redblackgraph/sparse/csgraph/_rbg_math.pxi` - Cython AVOS operations
- `redblackgraph/sparse/sparsetools/rbm_math.h` - C++ sparse operations

### Typical Workload Characteristics
- **Matrix size**: 15-5000 vertices (from notebooks/tests)
- **Sparsity**: 0.1%-5% (genealogy graphs: ~2-4 edges per vertex)
- **Operations**: Matrix multiplication, transitive closure, path finding
- **Data types**: int8, int16, int32, int64 (int32 most common)

## Success Criteria

1. **Correctness**: 100% agreement with CPU implementation (all 167 tests pass)
2. **Performance**: 5-50x speedup for matrices >1000x1000
3. **Usability**: Drop-in replacement for `rb_matrix` with `.gpu()` method
4. **Maintainability**: Clear, documented code following project standards
5. **Portability**: Works on NVIDIA GPUs (CUDA 11.0+)

## Getting Started

Read the documents in order:
1. Start with **Architecture** to understand the big picture
2. Review **CUDA Kernels** for implementation details
3. Check **CuPy Integration** for Python API design
4. Study **Performance Strategy** for optimization approach
5. Understand **Testing Plan** before implementing
6. Follow **Implementation Phases** for execution

## Notes on Original Plan

The original `gpu_implementation.md` was written before:
- NumPy 2.x migration
- Parity constraint refinements
- Mathematical analysis of non-associativity
- Addition of `RED_ONE`/`BLACK_ONE` constants

This updated plan supersedes the original but preserves its core insights on:
- CUDA kernel structure
- Sparse matrix strategies
- Two-pass CSR multiplication algorithm
- Performance optimization techniques
