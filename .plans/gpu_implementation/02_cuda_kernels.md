# CUDA kernels for avos SpGEMM (A × A, upper‑triangular)

We implement **two‑phase SpGEMM** with a structural **upper‑triangle mask** and a selectable strategy (**merge‑based** or **hash‑based**) for accumulation. Two execution modes are provided:

- **Deterministic:** ordered merges; no atomics; stable, bit‑exact output.
- **Fast:** per‑row hash accumulators with atomics; best throughput where nnz/row is moderate.

## Inputs/Outputs

- Input `A`: CSR (`indptrA`, `indicesA`, `dataA`), upper‑triangular.
- Output `C`: CSR (`indptrC`, `indicesC`, `dataC`), upper‑triangular.
- Semiring: `avos` (custom `add`, `mul`, identities, annihilator).
- Mask: `UpperTriangle` (`j >= i`).

## Phase 1 — Symbolic (pattern only)

Goal: compute `row_nnzC[i]` for all rows `i` so we can prefix‑sum into `indptrC` and allocate `indicesC`/`dataC`.

### Option A: Merge‑based (deterministic‑friendly)

For row `i`, each nonzero `A(i,k)` contributes row `k` of `A` into `C(i,:)`.

- Traverse candidate rows in **sorted column order**.
- Apply **mask** on the fly: skip any `j < i`.
- **Merge‑unique** the candidate columns (two‑pointer merges across sorted lists) to produce the **pattern** for row `i`. Count only unique `j`.

### Option B: Hash‑based (fast)

For row `i`, maintain a **row‑private hash set** `H` (warp/block‑local, with an overflow to global scratch).

- For each `A(i,k)`, iterate `A(k,:)` and insert `j` into `H` if `j >= i`.
- `row_nnzC[i] = |H|`.

**Load balancing:** assign short rows to single warps, long rows to full blocks (“row‑split”); optional cooperative ctas for pathological rows.

After symbolic, exclusive‑scan `row_nnzC` to build `indptrC` and allocate `indicesC` (and `dataC` for numeric).

## Phase 2 — Numeric (values)

Compute values with the same traversal order and mask as symbolic.

### Deterministic (merge‑based)

- Re‑run the same **ordered merges**, now performing avos operations:
  ```
  for i in rows:
      for each candidate column j (in ascending order):
          acc = add_identity
          # classic SpGEMM inner product with structural upper‑mask
          for k in intersection( row_i_of_A, column_j_of_A^T ):
              acc = add(acc, mul( A[i,k], A[k,j] ))
          C[i,j] = acc
  ```
- Because rows/columns are visited in a fixed order and no atomics are used, results are **bit‑exact** (modulo the semiring’s associativity).

### Fast (hash‑based)

- Use a **row‑private hash map** `M[j] -> value`:
  - Initialize `M` empty.
  - For each `A(i,k)`:
    - For each `A(k,j)` with `j >= i`:
      - `atomic_add(M[j], mul(A[i,k], A[k,j]))` using the semiring’s `add`.
  - Emit entries from `M` in sorted `j` order to `indicesC/dataC`.
- Provide a **fallback** for rows that overflow local hash capacity (spill to global workspace).

## Structural mask

Apply `j >= i` both:
- **During symbolic** (do not count masked columns), and
- **During numeric** (to guard against any residual masked writes).

## Index widths

- `indptrC`: **int64** when `n` or row counts are large; otherwise int32.
- `indicesC`: int32 if `n < 2^31`, else int64 (rare; prefer partitioning).

## Unified Memory (UVM) tuning

Even on GH200/H100:
- Prefetch CSR arrays to the active device before each phase: `cudaMemPrefetchAsync`.
- Mark large, read‑mostly arrays with `cudaMemAdviseSetReadMostly` and set preferred location to the device.
- Measure page‑faults and migrations (nsys / nvprof) and bake thresholds into perf CI.

## Pseudocode sketch (merge‑based symbolic)

```cpp
__global__ void symbolic_upper_merge(
    const int64_t* __restrict__ indptrA,
    const int32_t* __restrict__ indicesA,
    /* values optional here */,
    int64_t* __restrict__ row_nnzC,
    int64_t n_rows)
{
    // One row per warp or per block depending on length (omitted for brevity).
    int i = /* row assigned to this warp/block */;

    // Temporary per-row structure: either small fixed-size set or a streaming merge of sorted lists.
    // Apply mask: only j >= i survive.

    // 1) gather candidate lists: rows k in A reachable from row i
    // 2) merge-unique their sorted column lists, counting entries with j >= i
    // 3) write row_nnzC[i]
}
```

Numeric mirrors the traversal and writes `indicesC/dataC` in sorted order (deterministic) or uses row‑local hash accumulation (fast).


