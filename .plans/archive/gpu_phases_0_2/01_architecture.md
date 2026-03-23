# Architecture: Python API → CuPy → CUDA Kernels

This plan formalizes the layers and introduces first‑class **Semiring** and **Mask** abstractions so GPU and CPU paths share the same semantics.

## Layers

1. **Python API** (`rb_matrix_gpu`): mirrors the CPU API (constructor, `@` for avos matmul, conversions).
2. **CuPy layer**: uses `cupyx.scipy.sparse.csr_matrix` as the canonical GPU container and launches RawKernels.
3. **CUDA kernels**: implement two‑phase SpGEMM (symbolic → numeric) with an **upper‑triangle structural mask** and the **avos** semiring.

## SemiringSpec (avos)

Expose the algebra as data so kernels can be specialized and tests can assert invariants.

```python
class SemiringSpec(NamedTuple):
    # elementwise operators
    add: str   # name of device function for "addition"
    mul: str   # name of device function for "multiplication"
    add_identity: str  # literal or macro (e.g., "0")
    mul_identity: str  # literal or macro (e.g., "1")
    annihilator: str   # literal for multiplicative zero (e.g., "0")

    # algebraic properties (used to pick deterministic kernels)
    add_associative: bool
    add_commutative: bool
    add_idempotent: bool
    mul_associative: bool
    mul_commutative: bool
```

Provide a concrete `AVOS_SEMIRING` instance and export it from both CPU and GPU modules.

## MaskSpec

Make masking explicit and reusable:
- `UpperTriangle`: `mask(i,j) = (j >= i)`
- `Band(k)`: `mask(i,j) = (|j - i| <= k)`
- `GenerationLimit(g)`: struct. restriction by hop count (optional, future)

Masks are applied in **symbolic** (to bound pattern) and checked in **numeric**.

## Device policy

Surface a minimal policy so users/tests can **force CPU/GPU** and pick determinism:

- Environment variable: `RBG_DEVICE_POLICY={auto,cpu,gpu}`
- Context manager:
  ```python
  with rbg.device(policy="gpu", deterministic=True):
      C = A @ A  # avos
  ```

## Index widths policy

- `indices`: int32 by default (requires `n < 2^31`).
- `indptr`: int32 by default; **int64** when `n` or row counts can exceed 2^31‑1 (billion‑scale and/or heavy intermediate rows).
- Values: float32 or domain‑specific type (documented in SemiringSpec).


