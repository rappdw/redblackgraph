#!/usr/bin/env python3
"""
GPU Workflow Demo

Demonstrates the GPU-accelerated AVOS operations:
1. Creating CSRMatrixGPU from CPU sparse matrices
2. SpGEMM (sparse matrix multiply) on GPU
3. Transitive closure via repeated squaring on GPU
4. Round-trip CPU <-> GPU transfers

Run with: python examples/gpu_workflow.py
Requires: CuPy and an NVIDIA GPU
"""

import sys
import time
import numpy as np
from scipy.sparse import csr_matrix

try:
    from redblackgraph.gpu import CSRMatrixGPU, spgemm, transitive_closure_gpu
except ImportError:
    print("CuPy not available or no GPU detected. Install with: pip install cupy-cuda12x")
    sys.exit(1)


def demo_basic_operations():
    """Demonstrate basic GPU matrix operations."""
    print("=" * 60)
    print("Demo 1: Basic GPU Operations")
    print("=" * 60)
    print()

    # A small family DAG:
    #   0 (male)  -> 1 (female), 2 (male)
    #   1 (female) -> 3 (female)
    #   2 (male)  -> 3 (female)
    A = np.array([
        [ 1,  2,  3,  0],
        [ 0, -1,  0,  2],
        [ 0,  0,  1,  3],
        [ 0,  0,  0, -1],
    ], dtype=np.int32)

    print("Input adjacency matrix (upper triangular):")
    print(A)
    print()

    # Transfer to GPU
    A_cpu = csr_matrix(A)
    A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
    print(f"GPU matrix: {A_gpu}")
    print(f"Memory usage: {A_gpu.memory_usage()['total']} bytes")
    print()

    # SpGEMM: A @ A
    C_gpu = A_gpu @ A_gpu
    C_cpu = C_gpu.to_cpu().toarray()
    print("A @ A (AVOS multiply):")
    print(C_cpu)
    print()

    # Copy and compare
    B_gpu = A_gpu.copy()
    print(f"Copy: {B_gpu}")
    print()


def demo_transitive_closure():
    """Demonstrate GPU transitive closure."""
    print("=" * 60)
    print("Demo 2: Transitive Closure on GPU")
    print("=" * 60)
    print()

    # A longer chain: 0->1->2->3->4
    A = np.array([
        [ 1,  2,  0,  0,  0],
        [ 0, -1,  3,  0,  0],
        [ 0,  0,  1,  2,  0],
        [ 0,  0,  0, -1,  3],
        [ 0,  0,  0,  0,  1],
    ], dtype=np.int32)

    print("Input (5-vertex chain):")
    print(A)
    print()

    A_gpu = CSRMatrixGPU.from_cpu(csr_matrix(A), triangular=True)

    start = time.perf_counter()
    R_gpu, diameter = A_gpu.transitive_closure()
    elapsed = time.perf_counter() - start

    R = R_gpu.to_cpu().toarray()
    print(f"Transitive closure (computed in {elapsed:.4f}s):")
    print(R)
    print(f"Diameter estimate: {diameter}")
    print(f"Non-zeros: {A_gpu.nnz} -> {R_gpu.nnz}")
    print()


def demo_spgemm_general():
    """Demonstrate general A @ B multiplication."""
    print("=" * 60)
    print("Demo 3: General A @ B SpGEMM")
    print("=" * 60)
    print()

    A = np.array([
        [1, 2, 0],
        [0, -1, 3],
        [0, 0, 1],
    ], dtype=np.int32)

    B = np.array([
        [1, 0, 0, 2],
        [0, -1, 0, 0],
        [0, 0, 1, 3],
    ], dtype=np.int32)

    print("A (3x3):")
    print(A)
    print()
    print("B (3x4):")
    print(B)
    print()

    A_gpu = CSRMatrixGPU.from_cpu(csr_matrix(A))
    B_gpu = CSRMatrixGPU.from_cpu(csr_matrix(B))

    C_gpu = spgemm(A_gpu, B_gpu)
    C = C_gpu.to_cpu().toarray()
    print("C = A @ B (AVOS):")
    print(C)
    print()


def build_random_upper_triangular(n, edges_per_row=3, seed=42):
    """Build a random upper triangular sparse matrix."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(1 if i % 2 == 0 else -1)
        if i + 1 < n:
            for j in rng.choice(range(i + 1, n), size=min(edges_per_row, n - i - 1), replace=False):
                rows.append(i)
                cols.append(j)
                vals.append(int(rng.integers(2, 8)))

    return csr_matrix(
        (np.array(vals, dtype=np.int32), (rows, cols)),
        shape=(n, n)
    )


def demo_cpu_vs_gpu():
    """Compare GPU vs CPU performance across matrix sizes."""
    print("=" * 60)
    print("Demo 4: CPU vs GPU Performance Comparison")
    print("=" * 60)
    print()

    import cupy as cp
    from redblackgraph.sparse import rb_matrix
    from redblackgraph.sparse.csgraph.transitive_closure import transitive_closure_squaring
    from redblackgraph.gpu.spgemm import spgemm_with_stats

    sizes = [100, 200, 500, 1000, 2000]

    # Warm up GPU kernels (first call includes JIT compilation)
    print("Warming up GPU kernels (JIT compile)...")
    warmup = build_random_upper_triangular(50)
    warmup_gpu = CSRMatrixGPU.from_cpu(warmup, triangular=True)
    _ = warmup_gpu.transitive_closure()
    cp.cuda.Stream.null.synchronize()
    print()

    print(f"{'Size':>6s}  {'NNZ':>7s}  {'CPU TC':>10s}  {'GPU TC':>10s}  {'Speedup':>8s}  {'TC NNZ':>8s}")
    print("-" * 65)

    for n in sizes:
        A_cpu = build_random_upper_triangular(n, edges_per_row=3)

        # CPU transitive closure
        A_rb = rb_matrix(A_cpu.copy())
        start = time.perf_counter()
        tc_cpu = transitive_closure_squaring(A_rb)
        cpu_time = time.perf_counter() - start
        cpu_nnz = tc_cpu.W.nnz

        # GPU transitive closure
        A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        R_gpu, diameter = A_gpu.transitive_closure()
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"{n:>6d}  {A_cpu.nnz:>7d}  {cpu_time:>9.4f}s  {gpu_time:>9.4f}s  {speedup:>7.1f}x  {cpu_nnz:>8d}")

        # Verify correctness
        gpu_result = R_gpu.to_cpu().toarray()
        cpu_result = tc_cpu.W.toarray()
        assert np.array_equal(gpu_result, cpu_result), f"Mismatch at n={n}!"

    print()
    print("(All results verified bit-exact against CPU)")
    print()

    # Single-multiply SpGEMM breakdown for largest size
    n = sizes[-1]
    A_cpu = build_random_upper_triangular(n, edges_per_row=3)
    A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)

    print(f"SpGEMM breakdown for {n}x{n} (nnz={A_gpu.nnz}):")
    C_gpu, stats = spgemm_with_stats(A_gpu)
    cp.cuda.Stream.null.synchronize()
    print(f"  Symbolic: {stats.symbolic_time:.4f}s")
    print(f"  Numeric:  {stats.numeric_time:.4f}s")
    print(f"  Total:    {stats.total_time:.4f}s")
    print(f"  Output:   {stats.output_nnz} nnz")
    print()


if __name__ == "__main__":
    demo_basic_operations()
    demo_transitive_closure()
    demo_spgemm_general()
    demo_cpu_vs_gpu()
    print("All demos complete.")
