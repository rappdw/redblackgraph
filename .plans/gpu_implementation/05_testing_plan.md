# Testing plan

We require **bit‑exact** CPU↔GPU parity in **deterministic** mode and statistically robust coverage across sparsity regimes.

## Categories

1. **Golden parity** (small/medium): exact equality of CSR structure and values vs CPU for a matrix zoo (hand‑crafted + random).
2. **Property‑based** (Hypothesis):
   - Structural invariants (upper‑triangle holds; no masked entries).
   - Monotonicity/idempotence properties specific to avos (as applicable).
   - Adversarial rows (very large and very small row lengths).
3. **Stability**:
   - Deterministic mode: identical results across runs/devices.
   - Fast mode: acceptable numerical variance only if semiring permits (otherwise also exact).
4. **Index‑width**:
   - Threshold tests that flip `indptr` to int64 when predicted counts exceed 2^31‑1.
5. **Performance CI (nightly)**:
   - Guardrails on phase times, page‑fault counts (UVM), and workspace sizes.

## Test scaffolding

- A shared generator for upper‑triangular CSR matrices with tunable `n` and edges per node.
- Helpers to compare CSR structures efficiently (row‑by‑row diffs).
- Flags to select device policy and deterministic/fast mode from tests.

## Coverage-driven next steps (meaningful improvements)

Based on the current `pytest --cov=redblackgraph --cov-report=term-missing` report (overall ~73%):

1. **Exercise the real GPU dependency surface**
   - Add a small set of tests that verify the runtime environment is usable (NVRTC + cuBLAS + cuSPARSE loadable) *before* running heavier kernels.
   - Prefer tests that hit:
     - `cupyx.scipy.sparse` conversion paths
     - cuBLAS-backed dense helpers
     - custom RawModule compilation + caching

2. **Increase coverage in GPU edge/error paths**
   - `redblackgraph/gpu/core.py` and `redblackgraph/gpu/matrix.py` have meaningful untested branches around:
     - missing/disabled CuPy
     - shape mismatches
     - unsupported operations
   - Add explicit tests that assert the correct exception types/messages for these cases.

3. **Symbolic vs numeric phase invariants (SpGEMM)**
   - Add tests that check internal invariants that matter for correctness:
     - `indptrC[-1] == nnzC`
     - `indptrC` is monotone
     - `indicesC` are sorted within each row (if required)
     - triangular mask is preserved (no `j < i` entries)
   - Include adversarial patterns:
     - empty matrix
     - single-entry matrix
     - a row that produces a large candidate expansion (hash table sizing)

4. **Target the lowest-covered, highest-value CPU modules that support GPU workflows**
   - `redblackgraph/util/graph_builder.py` (~10%) is a high-impact integration point.
     - Add a minimal “happy path” integration test using a temp directory with a tiny relationship dataset and a stubbed crawler output.
     - Validate graph ordering selection and that GPU-capable paths can be selected when requested.

5. **Sparse csgraph helpers (validation + format handling)**
   - `_density.py` (~44%), `_sparse_format.py` (~55%), `transitive_closure.py` (~59%) are missing coverage primarily in:
     - validation/error branches
     - edge cases (empty graph, 1x1, non-square)
   - Add direct unit tests to hit these branches (don’t rely only on end-to-end closure tests).

6. **Test hygiene**
   - Register the `slow` marker in pytest config to avoid warning noise.
   - Consider splitting GPU tests into:
     - fast smoke tests (PR/CI)
     - large-matrix/perf tests (nightly)


