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


