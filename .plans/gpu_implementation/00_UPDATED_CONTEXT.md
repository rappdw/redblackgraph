# GPU Implementation — Updated Context

This document tightens scope and corrects billion‑scale arithmetic for the GPU acceleration of **avos** matrix multiplication in **redblackgraph**.

## Scope & invariants

- **Operation:** compute \(C = A \otimes A\) over the *avos* semiring and apply a **structural upper‑triangle mask** so that only entries with \(i \le j\) are produced in \(C\).
- **Input structure:** \(A\) is stored in **CSR**, assumed **upper‑triangular** (or made so by pre‑masking). We maintain that invariant across operations.
- **Single‑GPU first:** Target correctness and competitive performance on a single modern NVIDIA GPU (e.g., H100 80 GB or GH200). Multi‑GPU is a separate milestone.

## Correcting the 1B×1B, 0.1% density example

The earlier example estimated memory for a \(10^9 \times 10^9\) matrix at **0.1%** global density over the **upper triangle** as ~GB scale. That is incorrect by ~\(10^6\times\).

- Count of upper‑triangular positions: \(n(n+1)/2\). For \(n = 10^9\), that’s \(\approx 5\times10^{17}\).
- At **0.1%** density (\(10^{-3}\) fraction nonzero): \(\approx 5\times10^{14}\) nonzeros.
- CSR with 32‑bit indices and 32‑bit values uses ~**8 bytes per nnz** for `indices` + `data` (+ `indptr` overhead). That’s on the order of **4 petabytes** just for `indices` + `data`—infeasible for a single GPU or node.

### Practical framing: edges per node (O(n) nnz)

Genealogy‑style DAGs scale with **edges per node** \(e\) (typically small), not a fixed percent density. Let \(\text{nnz} \approx e\,n\). Then CSR memory is roughly:

- `indices` + `data`: \(8\,\text{bytes} \times \text{nnz}\) (for 32‑bit values; double if 64‑bit values).
- `indptr`: \((n+1) \times w\_\text{ptr}\) bytes, where \(w\_\text{ptr}\) is 4 bytes (int32) or 8 bytes (int64). Use **int64** if \(\text{nnz} \ge 2^{31}\) or if intermediate counts can overflow 32‑bit.

**Illustrative budgets (approx., CSR with 32‑bit `indices` and 32‑bit `data`):**

| n        | e (edges/node) | nnz        | `indices`+`data` | `indptr` (int32) | `indptr` (int64) | Total (int32 ptr) | Total (int64 ptr) |
|----------|-----------------|------------|------------------|------------------|------------------|-------------------|-------------------|
| 1e7      | 2               | 2.0e7      | ~160 MB          | ~40 MB           | ~80 MB           | ~200 MB           | ~240 MB           |
| 1e8      | 2               | 2.0e8      | ~1.6 GB          | ~0.4 GB          | ~0.8 GB          | ~2.0 GB           | ~2.4 GB           |
| 1e9      | 2               | 2.0e9      | ~16.0 GB         | ~4.0 GB          | ~8.0 GB          | ~20.0 GB          | ~24.0 GB          |
| 1e9      | 4               | 4.0e9      | ~32.0 GB         | ~4.0 GB          | ~8.0 GB          | ~36.0 GB          | ~40.0 GB          |

> **Takeaway:** Billion‑node problems are plausible on a single 80 GB GPU **if** nnz grows as \(O(n)\) with a small constant (and if intermediate SpGEMM growth is controlled via masking). Percent‑density framing is misleading for these graphs.

## Hardware & runtime assumptions

- **GPU:** H100 80 GB or GH200‑class. We will initially assume **single‑GPU** execution.
- **Unified Memory (UVM):** Use **UVM + prefetch + memory advice** for simplicity *and* performance. UVM makes correctness easier but still requires explicit tuning to avoid page‑fault thrash.
  - Use `cudaMemPrefetchAsync(ptr, bytes, device, stream)` before each phase.
  - Use `cudaMemAdvise` (e.g., `SetPreferredLocation`, `SetReadMostly`) for large, read‑mostly CSR arrays.
- **Data movement:** Keep inputs and outputs resident on device across phases (symbolic → numeric). Minimize host↔device synchronization in inner loops.

## Index widths

- **Default:** `indices` = **int32**, `indptr` = **int32** for small/medium cases.
- **Large scale:** Switch `indptr` to **int64** when \(n \ge 10^9\) or when symbolic estimates predict any row count or cumulative nnz could exceed 2^31‑1.
- Make index widths **per‑array** (it’s fine for `indices` to be int32 while `indptr` is int64).

## Structural mask (upper triangle)

- Treat upper‑triangularity as a **structural mask**: let `mask(i,j) = (j >= i)`. Apply the mask **in the symbolic phase** to cap the pattern up front and **again in numeric** (belt‑and‑suspenders).

## Terminology

- **avos**: project‑specific semiring (custom “add” and “mul” with identities and annihilator).
- **SpGEMM**: sparse‑sparse matrix multiply (two‑phase: symbolic → numeric).
- **CSR**: compressed sparse row storage (row pointer `indptr`, column indices `indices`, values `data`).


