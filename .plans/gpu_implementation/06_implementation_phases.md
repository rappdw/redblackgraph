# Implementation phases & acceptance criteria

**Current Status (Nov 6, 2025)**: Phases 0-2 complete, 62 tests passing  
**Implemented**: Production CSR matrix, AVOS kernels, symbolic + numeric SpGEMM  
**Next**: Phase 3 (hash-based numeric) or Phase 4 (UVM tuning) based on profiling

---

**Phase 0 — Baseline & scaffolding** ✅ **(COMPLETE)**
- [x] Introduce `CSRMatrixGPU` skeleton with raw int32 CSR buffers (production data structure).
- [x] CPU↔GPU conversion routines (`from_cpu()`, `to_cpu()`); unit tests (14 tests passing).
- [ ] Device policy surface (`RBG_DEVICE_POLICY` + context manager) - *deferred*.

**Phase 1 — Symbolic (pattern)** ✅ **(COMPLETE)**
- [x] Implement symbolic with **merge‑based** path; apply **upper‑triangle mask** (`spgemm_symbolic.py`).
- [x] Exclusive‑scan to form `indptrC` (`prefix_sum_scan()`); workspace sizing and error handling.
- [x] **Acceptance**: pattern matches CPU reference on test zoo (12 tests passing).
- [ ] Add **hash‑based** symbolic prototype - *deferred (merge-based sufficient)*.

**Phase 2 — Numeric (deterministic)** ✅ **(COMPLETE)**
- [x] Deterministic numeric via **bitmap accumulation** with AVOS operations (`spgemm_numeric.py`).
- [x] Emit `indicesC/dataC` in ascending column order to match CPU lexicographically.
- [x] **Acceptance**: **bit‑exact** equality vs CPU across test zoo (15 end-to-end tests, 62 total tests passing).

**Phase 3 — Numeric (fast)**
- [ ] Per‑row **hash** accumulators with atomics; overflow fallback path.
- [ ] Emission sorted by column to preserve CSR invariants.
- Acceptance: ≥X× speedup on mid‑sparsity workloads vs deterministic; still exact if semiring demands.

**Phase 3a — Triangularization (hybrid)** *(Optional, based on profiling)*
- [ ] GPU-accelerated permutation: `permute_gpu(A_gpu, perm)`.
- [ ] Hybrid workflow: CPU topological sort + GPU permutation application.
- [ ] High-level API: `triangularize_gpu(A_cpu, method='topological')`.
- Acceptance: 2-3× speedup on permutation step; bit-exact match with CPU.
- See: `07_triangularization.md` for detailed design.

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

