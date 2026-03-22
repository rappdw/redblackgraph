"""Benchmark CPU vs GPU transitive closure across graph sizes.

Compares three paths:
  1. CPU (auto)   — transitive_closure_squaring(), which auto-detects
                    DAGs and uses the O(V+E+nnz) Cython DAG algorithm
  2. GPU (sqr)    — repeated squaring via CUDA SpGEMM
  3. GPU (dag)    — level-parallel topological propagation on GPU
"""

import time
import sys
import numpy as np
from redblackgraph.util.synthesizer import FamilyDagSynthesizer, SynthesizerConfig
from redblackgraph.sparse.csgraph import transitive_closure_squaring

try:
    from redblackgraph.gpu import (
        CSRMatrixGPU, transitive_closure_gpu, transitive_closure_dag_gpu
    )
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
    repeats = 3 if n <= 5000 else 1

    print(f"\n--- Graph: {n:,} vertices, {nnz:,} nnz ---", flush=True)

    # CPU: auto-selects DAG algorithm for triangular matrices
    cpu_time = None
    if n <= 50_000:
        _ = transitive_closure_squaring(matrix)  # warmup
        times_cpu = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            transitive_closure_squaring(matrix)
            t1 = time.perf_counter()
            times_cpu.append(t1 - t0)
        cpu_time = min(times_cpu)

    gpu_matrix = CSRMatrixGPU.from_cpu(matrix)

    # GPU: repeated squaring
    _ = transitive_closure_gpu(gpu_matrix.copy())  # warmup
    times_sqr = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        transitive_closure_gpu(gpu_matrix.copy())
        t1 = time.perf_counter()
        times_sqr.append(t1 - t0)
    sqr_time = min(times_sqr)

    # GPU: DAG-specific level-parallel closure
    _ = transitive_closure_dag_gpu(gpu_matrix.copy())  # warmup
    times_dag = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        transitive_closure_dag_gpu(gpu_matrix.copy())
        t1 = time.perf_counter()
        times_dag.append(t1 - t0)
    dag_time = min(times_dag)

    cpu_str = f"{cpu_time:.4f}s" if cpu_time is not None else "skipped"
    best_gpu = min(sqr_time, dag_time)
    if cpu_time is not None:
        ratio = cpu_time / best_gpu
        speedup = f"{ratio:.1f}x GPU" if ratio >= 1 else f"{1/ratio:.1f}x CPU"
    else:
        speedup = "N/A"
    print(f"  CPU: {cpu_str}  GPU-Sqr: {sqr_time:.4f}s  GPU-DAG: {dag_time:.4f}s  ({speedup})",
          flush=True)

    results.append((n, nnz, cpu_time, sqr_time, dag_time))

# Print table
print("\n" + "=" * 96)
print(f"{'Vertices':>10} {'NNZ':>10} {'CPU-DAG (s)':>14} {'GPU-Sqr (s)':>14} "
      f"{'GPU-DAG (s)':>14} {'Best GPU/CPU':>14}")
print("-" * 96)
for n, nnz, cpu_t, sqr_t, dag_t in results:
    cpu_str = f"{cpu_t:.4f}" if cpu_t is not None else "—"
    best_gpu = min(sqr_t, dag_t)
    if cpu_t is not None:
        ratio = cpu_t / best_gpu
        if ratio >= 1:
            speedup_str = f"{ratio:.1f}x GPU"
        else:
            speedup_str = f"{1/ratio:.1f}x CPU"
    else:
        speedup_str = "—"
    print(f"{n:>10,} {nnz:>10,} {cpu_str:>14} {sqr_t:>14.4f} {dag_t:>14.4f} {speedup_str:>14}")
print("=" * 96)
print("\nCPU-DAG = Cython topological propagation, O(V+E+nnz_closure)")
print("GPU-Sqr = CUDA repeated squaring via SpGEMM, O(nnz² · log d)")
print("GPU-DAG = CUDA level-parallel topological propagation, O(V+E+nnz_closure)")
