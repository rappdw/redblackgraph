#!/usr/bin/env python3
"""
Naive GPU Implementation Demo

This script demonstrates the basic naive GPU implementation and helps you understand:
1. How to use the GPU module
2. Memory transfer patterns
3. Basic performance characteristics
4. Correctness validation against CPU reference

Run with: python examples/gpu_naive_demo.py
Requires: CuPy installed and GPU available
"""

import sys
import time
import numpy as np

# Check for GPU availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"✓ CuPy {cp.__version__} found")
    print(f"✓ GPU: {cp.cuda.Device(0).compute_capability}")
    print()
except ImportError:
    print("❌ CuPy not available. Install with: pip install cupy-cuda12x")
    print("   (or cupy-cuda11x for CUDA 11.x)")
    sys.exit(1)

from redblackgraph.gpu import avos_sum_gpu, avos_product_gpu, rb_matrix_gpu
from redblackgraph.reference.rbg_math import avos_sum, avos_product


def demo_element_wise_operations():
    """Demonstrate basic AVOS operations on GPU."""
    print("=" * 60)
    print("Demo 1: Element-wise AVOS Operations")
    print("=" * 60)
    print()
    
    # Test cases covering various scenarios
    test_cases = [
        ("Basic sum", 3, 5, "avos_sum"),
        ("Basic product", 2, 3, "avos_product"),
        ("RED_ONE identity", -1, -1, "avos_product"),
        ("BLACK_ONE identity", 1, 1, "avos_product"),
        ("Parity filter (even)", 2, -1, "avos_product"),
        ("Parity filter (odd)", 3, 1, "avos_product"),
    ]
    
    for name, x, y, op in test_cases:
        # CPU reference
        if op == "avos_sum":
            cpu_result = avos_sum(x, y)
        else:
            cpu_result = avos_product(x, y)
        
        # GPU computation
        x_gpu = cp.array([x], dtype=cp.int32)
        y_gpu = cp.array([y], dtype=cp.int32)
        
        if op == "avos_sum":
            gpu_result_array = avos_sum_gpu(x_gpu, y_gpu)
        else:
            gpu_result_array = avos_product_gpu(x_gpu, y_gpu)
        
        gpu_result = gpu_result_array.get()[0]
        
        # Validate
        match = "✓" if cpu_result == gpu_result else "✗"
        print(f"{match} {name:25s}: {op}({x:3d}, {y:3d}) = {gpu_result:3d} (CPU: {cpu_result:3d})")
    
    print()


def demo_vectorized_operations():
    """Demonstrate vectorized operations on arrays."""
    print("=" * 60)
    print("Demo 2: Vectorized Operations")
    print("=" * 60)
    print()
    
    # Create arrays
    size = 1000
    x_cpu = np.random.randint(2, 100, size=size, dtype=np.int32)
    y_cpu = np.random.randint(2, 100, size=size, dtype=np.int32)
    
    print(f"Processing {size} elements...")
    print()
    
    # CPU timing
    start = time.time()
    sum_cpu = np.array([avos_sum(x_cpu[i], y_cpu[i]) for i in range(size)], dtype=np.int32)
    cpu_time = time.time() - start
    
    # GPU timing (including transfer)
    start = time.time()
    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)
    cp.cuda.Stream.null.synchronize()  # Wait for transfer
    
    gpu_compute_start = time.time()
    sum_gpu = avos_sum_gpu(x_gpu, y_gpu)
    cp.cuda.Stream.null.synchronize()  # Wait for computation
    gpu_compute_time = time.time() - gpu_compute_start
    
    sum_result = sum_gpu.get()  # Transfer back
    gpu_time = time.time() - start
    
    # Validate
    matches = np.sum(sum_cpu == sum_result)
    
    print(f"CPU time:                {cpu_time*1000:.3f} ms")
    print(f"GPU time (total):        {gpu_time*1000:.3f} ms")
    print(f"GPU time (compute only): {gpu_compute_time*1000:.3f} ms")
    print(f"Speedup (compute only):  {cpu_time/gpu_compute_time:.1f}x")
    print(f"Correctness:             {matches}/{size} matches")
    
    if matches == size:
        print("✓ All results match CPU reference")
    else:
        print(f"✗ {size - matches} mismatches found")
    
    print()


def demo_sparse_matrix():
    """Demonstrate sparse matrix operations."""
    print("=" * 60)
    print("Demo 3: Sparse Matrix Operations")
    print("=" * 60)
    print()
    
    from scipy import sparse as sp_sparse
    
    # Create small sparse matrix
    size = 5
    data = np.array([2, 3, 4, 5, 6], dtype=np.int32)
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    
    A_cpu = sp_sparse.csr_matrix((data, indices, indptr), shape=(size, size))
    
    print(f"Matrix shape: {A_cpu.shape}")
    print(f"Non-zeros:    {A_cpu.nnz}")
    print(f"Density:      {A_cpu.nnz / (size * size) * 100:.1f}%")
    print()
    
    # Transfer to GPU
    print("Transferring to GPU...")
    start = time.time()
    A_gpu = rb_matrix_gpu.from_cpu(A_cpu, triangular=True)
    transfer_time = time.time() - start
    
    print(f"✓ Transfer complete in {transfer_time*1000:.3f} ms")
    print(f"  {A_gpu}")
    print()
    
    # Transfer back
    print("Transferring back to CPU...")
    start = time.time()
    A_result = A_gpu.to_cpu()
    back_time = time.time() - start
    
    print(f"✓ Transfer complete in {back_time*1000:.3f} ms")
    
    # Validate
    if np.array_equal(A_cpu.data, A_result.data) and \
       np.array_equal(A_cpu.indices, A_result.indices) and \
       np.array_equal(A_cpu.indptr, A_result.indptr):
        print("✓ CPU ↔ GPU round trip successful")
    else:
        print("✗ Data mismatch after round trip")
    
    print()


def demo_memory_patterns():
    """Demonstrate memory allocation and transfer patterns."""
    print("=" * 60)
    print("Demo 4: Memory Management")
    print("=" * 60)
    print()
    
    # Check memory before
    mempool = cp.get_default_memory_pool()
    used_before = mempool.used_bytes()
    total_before = mempool.total_bytes()
    
    print(f"Memory before:")
    print(f"  Used:  {used_before / 1024**2:.2f} MB")
    print(f"  Total: {total_before / 1024**2:.2f} MB")
    print()
    
    # Allocate arrays
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        # Allocate
        x = cp.arange(size, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        
        used = mempool.used_bytes()
        allocated_mb = (size * 4) / 1024**2  # int32 = 4 bytes
        
        print(f"Allocated {size:>7d} elements ({allocated_mb:>6.2f} MB)")
        print(f"  Memory used: {used / 1024**2:.2f} MB")
        
        # Free
        del x
        mempool.free_all_blocks()
    
    print()
    
    # On Grace Hopper (DGX Spark), this would use unified memory
    print("Note: On DGX Spark (Grace Hopper), CuPy can use unified")
    print("      memory for automatic CPU/GPU access without explicit transfers.")
    print()


def demo_performance_note():
    """Display performance notes and next steps."""
    print("=" * 60)
    print("Performance Notes")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT: This is a NAIVE implementation for learning only!")
    print()
    print("Current limitations:")
    print("  • Element-wise operations: Optimized (using CuPy kernels)")
    print("  • Matrix multiplication: NAIVE O(n³) with dense conversion")
    print("  • Memory usage: Not optimized for large matrices")
    print("  • Unified memory: Not yet leveraged on Grace Hopper")
    print()
    print("For production use, implement optimized kernels:")
    print("  1. Two-phase SpGEMM (symbolic + numeric)")
    print("  2. Upper triangular optimization (50% savings)")
    print("  3. Unified memory prefetching")
    print("  4. Custom hash-based accumulation")
    print()
    print("See .plans/gpu_implementation/ for full implementation plan.")
    print()


def main():
    """Run all demos."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Naive GPU Implementation Demo".center(58) + "║")
    print("║" + "  redblackgraph - AVOS GPU Operations".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        demo_element_wise_operations()
        demo_vectorized_operations()
        demo_sparse_matrix()
        demo_memory_patterns()
        demo_performance_note()
        
        print("=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review docs/gpu_naive_implementation.md")
        print("  2. Run tests: pytest tests/gpu/test_naive_gpu.py -v")
        print("  3. Read .plans/gpu_implementation/QUICK_START.md")
        print("  4. Begin implementing optimized CUDA kernels")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
