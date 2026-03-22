"""Benchmark CPU vs GPU transitive closure across graph sizes."""

import time
import sys
import numpy as np
from redblackgraph.util.synthesizer import FamilyDagSynthesizer, SynthesizerConfig
from redblackgraph.sparse.csgraph import transitive_closure_squaring

try:
    from redblackgraph.gpu import CSRMatrixGPU, transitive_closure_gpu
    HAS_GPU = True
except Exception as e:
    print(f"GPU not available: {e}")
    sys.exit(1)

SIZES = [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]


def make_graph(target, seed=42):
    """Generate a family DAG near the target vertex count."""
    if target <= 100:
        n0, gens = target, 0
    else:
        n0 = max(20, target // 20)
        gens = 6
    config = SynthesizerConfig(
        num_initial_nodes=n0,
        pct_red=50.0,
        avg_children_per_pairing=2.5,
        num_generations=gens,
        pct_monogamous=60.0,
        pct_non_procreating=15.0,
        seed=seed,
        max_total_vertices=target,
    )
    return FamilyDagSynthesizer(config).synthesize()


results = []

for target in SIZES:
    synth_result = make_graph(target)
    matrix = synth_result.matrix
    n = matrix.shape[0]
    nnz = matrix.nnz

    print(f"\n--- Graph: {n:,} vertices, {nnz:,} nnz ---", flush=True)

    # CPU: repeated squaring (same algorithm as GPU for fair comparison)
    cpu_time = None
    if n <= 25_000:
        # Warm up
        _ = transitive_closure_squaring(matrix)
        times_cpu = []
        repeats = 3 if n <= 2000 else 1
        for _ in range(repeats):
            t0 = time.perf_counter()
            tc_cpu = transitive_closure_squaring(matrix)
            t1 = time.perf_counter()
            times_cpu.append(t1 - t0)
        cpu_time = min(times_cpu)

    # GPU
    gpu_matrix = CSRMatrixGPU.from_cpu(matrix)
    # Warm up
    _ = transitive_closure_gpu(gpu_matrix.copy())

    times_gpu = []
    repeats = 3 if n <= 5000 else 1
    for _ in range(repeats):
        gpu_m = gpu_matrix.copy()
        t0 = time.perf_counter()
        tc_gpu, diam = transitive_closure_gpu(gpu_m)
        t1 = time.perf_counter()
        times_gpu.append(t1 - t0)
    gpu_time = min(times_gpu)

    cpu_str = f"{cpu_time:.4f}s" if cpu_time is not None else "skipped"
    if cpu_time is not None:
        ratio = cpu_time / gpu_time
        speedup = f"{ratio:.1f}x GPU" if ratio >= 1 else f"{1/ratio:.1f}x CPU"
    else:
        speedup = "N/A"
    print(f"  CPU: {cpu_str}  GPU: {gpu_time:.4f}s  ({speedup})", flush=True)

    results.append((n, nnz, cpu_time, gpu_time))

# Print table
print("\n" + "=" * 72)
print(f"{'Vertices':>10} {'NNZ':>10} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
print("-" * 72)
for n, nnz, cpu_t, gpu_t in results:
    cpu_str = f"{cpu_t:.4f}" if cpu_t is not None else "—"
    if cpu_t is not None:
        ratio = cpu_t / gpu_t
        if ratio >= 1:
            speedup_str = f"{ratio:.1f}x GPU"
        else:
            speedup_str = f"{1/ratio:.1f}x CPU"
    else:
        speedup_str = "—"
    print(f"{n:>10,} {nnz:>10,} {cpu_str:>12} {gpu_t:>12.4f} {speedup_str:>10}")
print("=" * 72)
print("\nCPU = sparse repeated squaring (Cython/SciPy)")
print("GPU = sparse repeated squaring (CUDA SpGEMM)")
print("Crossover point: GPU faster when Speedup shows 'GPU'")
