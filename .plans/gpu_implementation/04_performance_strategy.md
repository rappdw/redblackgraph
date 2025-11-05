# Performance strategy

We ground performance work in a small set of repeatable experiments and a roofline‑style model.

## Workloads

- Synthetic CSR matrices with:
  - `n ∈ {1e6, 1e7, 1e8}` (and scaled‑down shapes for CI)
  - edges per node `e ∈ {1,2,4,8}`
  - controlled row‑length variance (uniform vs. power‑law)
- Realistic graph‑like patterns if available (optional).

## Metrics

- Wall time for `C = A @ A` (avos) split by **symbolic** and **numeric**.
- Effective bandwidth (GB/s) in symbolic; flops‑like units in numeric (semiring‑dependent).
- Peak memory and temporary workspace size.
- Page‑fault/migration counts under UVM.

## Experiments

1. **Deterministic vs fast**: report speedup and exactness.
2. **Merge‑ vs hash‑based**: pick the winner per sparsity regime.
3. **UVM on/off**: with/without explicit `cudaMemPrefetchAsync` and `cudaMemAdvise`.
4. **cuSPARSE baseline**: compute A×A under (+,*) to ensure we’re within a reasonable factor of vendor SpGEMM (sanity check only; avos uses custom semiring).

## Roofline (back‑of‑envelope)

- Symbolic is memory‑bound; target high L2 hit rate and coalesced reads of `indices`.
- Numeric may be memory‑bound or latency‑bound depending on arithmetic intensity of `avos.mul`/`avos.add`.
- Use Nsight Systems/Compute to confirm and to tune block sizes and staging thresholds.


