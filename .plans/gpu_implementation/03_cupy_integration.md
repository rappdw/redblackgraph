# CuPy integration

We use `cupyx.scipy.sparse.csr_matrix` as the canonical GPU container and expose a Python API aligned with the CPU path.

## Public API surface

```python
def to_gpu(A_cpu: rb_matrix) -> rb_matrix_gpu: ...
def to_cpu(A_gpu: rb_matrix_gpu) -> rb_matrix: ...

class rb_matrix_gpu:
    def __matmul__(self, other: "rb_matrix_gpu") -> "rb_matrix_gpu":  # avos
        ...

    @property
    def csr(self) -> cupyx.scipy.sparse.csr_matrix: ...

    def astype(self, dtype): ...
```

- The `@` operator performs **avos** matmul via the CUDA kernels.
- Conversions ensure structural upper‑triangle invariants are preserved.

## RawKernel packaging

- Provide RawKernel sources with `#define`s controlled by `SemiringSpec` flags.
- Cache compiled binaries per `sm` architecture to avoid first‑use jitter.
- Consider shipping precompiled cubins for common archs (sm_80, sm_90).

## Interop

- **DLPack** bridges to exchange device arrays with PyTorch/JAX if needed.
- **Zero‑copy** between CuPy and kernels (no intermediate host copies).

## Error handling

- Validate semiring assumptions at import time (e.g., add associativity expected for deterministic mode).
- Raise informative errors when index‑width requirements are violated (e.g., int64 `indptr` required).


