#!/usr/bin/env python3
"""
Sparse Workflow Demo

This script demonstrates the sparse-only transitive closure workflow for large DAGs.
The sparse DAG closure algorithm never allocates O(N^2) memory, making it suitable
for very large graphs where memory is a constraint.

Run with: python examples/sparse_workflow.py
"""

import sys
import time
import numpy as np
from scipy.sparse import csr_matrix


def demo_sparse_dag_closure():
    """Demonstrate sparse DAG transitive closure."""
    print("=" * 60)
    print("Demo 1: Sparse DAG Transitive Closure")
    print("=" * 60)
    print()
    
    from redblackgraph.sparse import rb_matrix
    from redblackgraph.sparse.csgraph.transitive_closure import (
        transitive_closure_dag_sparse,
        transitive_closure_adaptive,
    )
    
    # Create a simple DAG (upper triangular)
    # Graph: 0 -> 1 -> 2 -> 3
    #        0 -> 2
    data = np.array([1, 2, 1, 4, 1, 8, 1], dtype=np.int32)
    indices = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    indptr = np.array([0, 2, 4, 6, 7], dtype=np.int32)
    
    A = rb_matrix((data, indices, indptr), shape=(4, 4))
    
    print("Input DAG (4 vertices):")
    print(f"  Edges: 0->1 (val=2), 0->2 (val=4), 1->2 (val=4), 2->3 (val=8)")
    print(f"  Non-zeros: {A.nnz}")
    print()
    
    # Compute sparse closure
    print("Computing transitive closure (sparse-only mode)...")
    start = time.time()
    result = transitive_closure_dag_sparse(A)
    elapsed = time.time() - start
    
    print(f"  Duration: {elapsed*1000:.3f} ms")
    print(f"  Closure non-zeros: {result.W.nnz}")
    print(f"  Diameter: {result.diameter}")
    print()
    
    # Show closure matrix
    print("Closure matrix (dense view for small example):")
    print(result.W.toarray())
    print()
    
    # Verify against adaptive method
    print("Verifying against standard transitive closure...")
    result_standard = transitive_closure_adaptive(A, method="FW")
    
    # Compare
    diff = result.W.toarray() - result_standard.W
    if np.allclose(diff, 0):
        print("  Results match standard Floyd-Warshall")
    else:
        print("  WARNING: Results differ from standard method")
        print(f"  Max difference: {np.max(np.abs(diff))}")
    print()


def demo_sparse_only_mode():
    """Demonstrate sparse_only parameter in transitive_closure_adaptive."""
    print("=" * 60)
    print("Demo 2: Using sparse_only Mode")
    print("=" * 60)
    print()
    
    from redblackgraph.sparse import rb_matrix
    from redblackgraph.sparse.csgraph.transitive_closure import transitive_closure_adaptive
    from redblackgraph.sparse.csgraph._density import DensificationError
    
    # Create a larger DAG
    n = 100
    print(f"Creating random DAG with {n} vertices...")
    
    # Create upper triangular matrix (guaranteed DAG)
    np.random.seed(42)
    density = 0.1
    
    rows = []
    cols = []
    data = []
    
    # Add identity diagonal
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(1 if i % 2 else -1)  # Alternating BLACK_ONE/RED_ONE
    
    # Add random upper triangular edges
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < density:
                rows.append(i)
                cols.append(j)
                data.append(2 ** np.random.randint(1, 5))  # Random AVOS values
    
    A = rb_matrix(
        csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int32)
    )
    
    print(f"  Non-zeros: {A.nnz}")
    print(f"  Density: {A.nnz / (n * n) * 100:.2f}%")
    print()
    
    # Compute with sparse_only=True
    print("Computing transitive closure with sparse_only=True...")
    start = time.time()
    result = transitive_closure_adaptive(A, sparse_only=True)
    elapsed = time.time() - start
    
    print(f"  Duration: {elapsed*1000:.3f} ms")
    print(f"  Closure non-zeros: {result.W.nnz}")
    print(f"  Closure density: {result.W.nnz / (n * n) * 100:.2f}%")
    print(f"  Diameter: {result.diameter}")
    print()
    
    # Demonstrate error on cyclic graph
    print("Demonstrating error handling for cyclic graphs...")
    
    # Create a graph with a cycle: 0 -> 1 -> 2 -> 0
    cycle_data = np.array([1, 2, 1, 4, 1, 8], dtype=np.int32)
    cycle_indices = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)
    cycle_indptr = np.array([0, 2, 4, 6], dtype=np.int32)
    
    A_cycle = rb_matrix((cycle_data, cycle_indices, cycle_indptr), shape=(3, 3))
    
    try:
        transitive_closure_adaptive(A_cycle, sparse_only=True)
        print("  ERROR: Should have raised DensificationError")
    except DensificationError as e:
        print(f"  Correctly raised DensificationError: {str(e)[:60]}...")
    print()


def demo_memory_comparison():
    """Compare memory usage between sparse and dense approaches."""
    print("=" * 60)
    print("Demo 3: Memory Usage Comparison")
    print("=" * 60)
    print()
    
    # Show theoretical memory usage for different graph sizes
    sizes = [100, 1000, 10000, 100000]
    
    print("Theoretical memory usage comparison:")
    print()
    print(f"{'Vertices':>10} | {'Dense O(N^2)':>15} | {'Sparse (10% density)':>20}")
    print("-" * 50)
    
    for n in sizes:
        dense_bytes = n * n * 4  # int32
        sparse_bytes = int(n * n * 0.1 * (4 + 4 + 4/n))  # data + indices + indptr overhead
        
        dense_str = format_bytes(dense_bytes)
        sparse_str = format_bytes(sparse_bytes)
        
        print(f"{n:>10,} | {dense_str:>15} | {sparse_str:>20}")
    
    print()
    print("Note: Sparse-only mode guarantees no O(N^2) allocations,")
    print("      making it feasible to process graphs with millions of vertices")
    print("      on systems with limited memory.")
    print()


def format_bytes(n):
    """Format byte count as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def demo_canonical_ordering():
    """Demonstrate canonical ordering with sparse closure."""
    print("=" * 60)
    print("Demo 4: Canonical Ordering with Sparse Closure")
    print("=" * 60)
    print()
    
    from redblackgraph.sparse import rb_matrix
    from redblackgraph.sparse.csgraph.transitive_closure import transitive_closure_dag_sparse
    from redblackgraph.sparse.csgraph import avos_canonical_ordering
    
    # Create a small DAG
    # Graph structure:
    #   0 -> 1
    #   0 -> 2
    #   1 -> 3
    #   2 -> 3
    data = np.array([1, 2, 4, 1, 8, 1, 8, 1], dtype=np.int32)
    indices = np.array([0, 1, 2, 1, 3, 2, 3, 3], dtype=np.int32)
    indptr = np.array([0, 3, 5, 7, 8], dtype=np.int32)
    
    A = rb_matrix((data, indices, indptr), shape=(4, 4))
    
    print("Input DAG:")
    print("  0 -> 1, 0 -> 2")
    print("  1 -> 3, 2 -> 3")
    print()
    
    # Compute sparse closure
    print("Step 1: Compute transitive closure (sparse-only)...")
    closure = transitive_closure_dag_sparse(A)
    print(f"  Closure computed: {closure.W.nnz} non-zeros")
    print()
    
    # Compute canonical ordering
    print("Step 2: Compute canonical ordering...")
    ordering = avos_canonical_ordering(closure.W)
    print(f"  Permutation: {ordering.label_permutation}")
    print(f"  Components: {ordering.components}")
    print()
    
    print("Canonical matrix (upper triangular):")
    print(ordering.A.toarray())
    print()


def main():
    """Run all demos."""
    print()
    print("=" * 60)
    print("  Sparse Workflow Demo")
    print("  redblackgraph - Memory-Efficient Transitive Closure")
    print("=" * 60)
    print()
    
    try:
        demo_sparse_dag_closure()
        demo_sparse_only_mode()
        demo_memory_comparison()
        demo_canonical_ordering()
        
        print("=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print()
        print("Key takeaways:")
        print("  1. transitive_closure_dag_sparse() never allocates O(N^2) memory")
        print("  2. Use sparse_only=True in transitive_closure_adaptive() for DAGs")
        print("  3. Use --sparse-only flag in scripts/compute_canonical_forms.py")
        print("  4. Cyclic graphs will raise DensificationError in sparse-only mode")
        print()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
