# Implementation phases & acceptance criteria

**Phase 0 — Baseline & scaffolding**
- [ ] Introduce `rb_matrix_gpu` skeleton, `cupyx.scipy.sparse.csr_matrix` container, and device policy surface (`RBG_DEVICE_POLICY` + context manager).
- [ ] CPU↔GPU conversion routines; unit tests.

**Phase 1 — Symbolic (pattern)**
- [ ] Implement symbolic with **merge‑based** path; apply **upper‑triangle mask**.
- [ ] Add **hash‑based** symbolic prototype (optional if merge‑based suffices).
- [ ] Exclusive‑scan to form `indptrC`; workspace sizing and error handling.
- Acceptance: pattern matches CPU reference on test zoo.

**Phase 2 — Numeric (deterministic)**
- [ ] Deterministic numeric via **ordered merges**; no atomics.
- [ ] Emit `indicesC/dataC` in ascending column order to match CPU lexicographically.
- Acceptance: **bit‑exact** equality vs CPU across test zoo.

**Phase 3 — Numeric (fast)**
- [ ] Per‑row **hash** accumulators with atomics; overflow fallback path.
- [ ] Emission sorted by column to preserve CSR invariants.
- Acceptance: ≥X× speedup on mid‑sparsity workloads vs deterministic; still exact if semiring demands.

**Phase 4 — UVM tuning & perf**
- [ ] Add `cudaMemPrefetchAsync` and `cudaMemAdvise` options (on by default).
- [ ] Perf harness with split timings (symbolic/numeric) and Nsight markers.
- Acceptance: documented improvements; thresholds captured in perf CI.

**Phase 5 — Packaging & docs**
- [ ] RawKernel caching by `sm` arch; optional precompiled cubins.
- [ ] User docs for semiring/mask, device policy, and index‑width rules.
- Acceptance: wheels build locally; README updated.

**Phase 6 — Multi‑GPU (deferred)**
- [ ] Evaluate 1D vs 2D partitioning; NVLink topology constraints.
- [ ] Decide go/no‑go based on single‑GPU results and product needs.

