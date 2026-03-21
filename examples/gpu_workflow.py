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


def demo_performance():
    """Demonstrate SpGEMM with timing on a larger matrix."""
    print("=" * 60)
    print("Demo 4: Performance (larger matrix)")
    print("=" * 60)
    print()

    from redblackgraph.gpu.spgemm import spgemm_with_stats

    n = 200
    # Build a random upper triangular sparse matrix
    rng = np.random.default_rng(42)
    rows, cols, vals = [], [], []
    for i in range(n):
        # Diagonal
        rows.append(i)
        cols.append(i)
        vals.append(1 if i % 2 == 0 else -1)
        # A few random edges above diagonal
        for j in rng.choice(range(i + 1, n), size=min(3, n - i - 1), replace=False):
            rows.append(i)
            cols.append(j)
            vals.append(int(rng.integers(2, 8)))

    A_cpu = csr_matrix(
        (np.array(vals, dtype=np.int32), (rows, cols)),
        shape=(n, n)
    )
    A_gpu = CSRMatrixGPU.from_cpu(A_cpu, triangular=True)
    print(f"Matrix: {n}x{n}, nnz={A_gpu.nnz}")

    C_gpu, stats = spgemm_with_stats(A_gpu)
    print(f"SpGEMM stats: {stats}")
    print(f"  Symbolic: {stats.symbolic_time:.4f}s")
    print(f"  Numeric:  {stats.numeric_time:.4f}s")
    print(f"  Output:   {stats.output_nnz} nnz")
    print()

    start = time.perf_counter()
    R_gpu, diameter = A_gpu.transitive_closure()
    elapsed = time.perf_counter() - start
    print(f"Transitive closure: {elapsed:.4f}s, nnz={R_gpu.nnz}, diameter={diameter}")
    print()


if __name__ == "__main__":
    demo_basic_operations()
    demo_transitive_closure()
    demo_spgemm_general()
    demo_performance()
    print("All demos complete.")
