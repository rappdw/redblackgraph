# CUDA Kernel Specifications

## Overview

This document details the CUDA kernel implementations for AVOS operations, including the asymmetric identity behavior and parity constraints introduced in the NumPy 2.x era.

## Core AVOS Operations

### 1. AVOS Sum Kernel

**Mathematical Definition**: Non-zero minimum
```
avos_sum(x, y) = {
  y,           if x == 0
  x,           if y == 0  
  min(x, y),   otherwise
}
```

**CUDA Implementation**:

```cuda
template<typename T>
__device__ T avos_sum(T x, T y) {
    if (x == 0) return y;
    if (y == 0) return x;
    
    // For signed types, compare as unsigned to handle negative values correctly
    using UT = typename std::make_unsigned<T>::type;
    return (static_cast<UT>(x) < static_cast<UT>(y)) ? x : y;
}

template<typename T>
__global__ void avos_sum_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = avos_sum(x[idx], y[idx]);
    }
}
```

**Characteristics**:
- **Complexity**: O(1) per element
- **Memory access**: 2 reads, 1 write per thread
- **Branching**: 2 branches (well-predicted for sparse data)
- **Coalescing**: Perfect if arrays are aligned

### 2. Most Significant Bit (MSB) Helper

**Mathematical Definition**: Bit position of leftmost 1-bit

```cuda
template<typename T>
__device__ int msb_position(T x) {
    if (x <= 1) return 0;
    
    // Use CUDA intrinsics for efficient bit scanning
    using UT = typename std::make_unsigned<T>::type;
    UT ux = static_cast<UT>(x);
    
    // __clz/__clzll returns count of leading zeros
    // MSB position = (bits - 1) - leading_zeros
    int leading_zeros;
    if (sizeof(T) == 8) {
        leading_zeros = __clzll(ux);
        return 63 - leading_zeros;
    } else if (sizeof(T) == 4) {
        leading_zeros = __clz(static_cast<uint32_t>(ux));
        return 31 - leading_zeros;
    } else {
        // For int8/int16, promote to int32
        leading_zeros = __clz(static_cast<uint32_t>(ux));
        return 31 - leading_zeros - (32 - sizeof(T) * 8);
    }
}
```

### 3. AVOS Product Kernel with Parity Constraints

**Mathematical Definition**: Bit-shift with asymmetric identity behavior

```cuda
template<typename T>
__device__ T avos_product(T lhs, T rhs) {
    using UT = typename std::make_unsigned<T>::type;
    UT x = static_cast<UT>(lhs);
    UT y = static_cast<UT>(rhs);
    
    // Zero property
    if (x == 0 || y == 0) {
        return 0;
    }
    
    // Constants for RED_ONE and BLACK_ONE
    const T RED_ONE = static_cast<T>(-1);
    const T BLACK_ONE = static_cast<T>(1);
    
    // Identity ⊗ Identity special cases (same-gender self-loops)
    if (lhs == RED_ONE && rhs == RED_ONE) {
        return x;  // RED_ONE ⊗ RED_ONE = RED_ONE
    }
    if (lhs == BLACK_ONE && rhs == BLACK_ONE) {
        return BLACK_ONE;  // BLACK_ONE ⊗ BLACK_ONE = BLACK_ONE
    }
    
    // Cross-gender identity cases (undefined relationships)
    if (lhs == RED_ONE && rhs == BLACK_ONE) {
        return 0;  // male's female self is undefined
    }
    if (lhs == BLACK_ONE && rhs == RED_ONE) {
        return 0;  // female's male self is undefined
    }
    
    // Identity on LEFT (lhs): Starting point marker
    // Treat RED_ONE as 1 for bit-shift composition
    if (lhs == RED_ONE) {
        x = 1;
    }
    
    // Identity on RIGHT (rhs): Gender/parity filter
    
    // When rhs is RED_ONE: Filters for even values only
    if (rhs == RED_ONE) {
        if (lhs & 1) {  // lhs is odd (use original lhs)
            return 0;   // Odd values have no male self
        } else {
            return x;   // Even values' male self is themselves
        }
    }
    
    // When rhs is BLACK_ONE: Filters for odd values only  
    if (rhs == BLACK_ONE) {
        if (lhs & 1) {  // lhs is odd
            return x;   // Odd values' female self is themselves
        } else {
            return 0;   // Even values have no female self
        }
    }
    
    // Normal case: bit-shift operation
    // avos_product(x, y) = (y & mask) | (x << bit_pos)
    // where mask = 2^bit_pos - 1
    int bit_position = msb_position(y);
    UT mask = (static_cast<UT>(1) << bit_position) - 1;
    
    return static_cast<T>((y & mask) | (x << bit_position));
}

template<typename T>
__global__ void avos_product_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = avos_product(x[idx], y[idx]);
    }
}
```

**Characteristics**:
- **Complexity**: O(1) per element (constant time bit operations)
- **Branching**: Multiple branches for identity/parity checks
  - Well-predicted for genealogy (identities are rare)
  - Divergence minimal in practice
- **Registers**: ~12-15 per thread (estimated)
- **Occupancy**: Should achieve 50%+ occupancy

### Performance Notes on Parity Constraints

The parity checks add ~6 conditional branches:
1. Check if lhs or rhs is zero (2 branches)
2. Check identity ⊗ identity cases (4 branches)
3. Check identity on left (1 branch)
4. Check identity on right (2 branches)

**Branch prediction**: Modern GPUs (Ampere+) have good branch prediction. For genealogy workloads:
- Identities appear infrequently (~0.1-1% of elements)
- Most threads take the "normal case" path
- Warp divergence is minimal

**Alternative**: Could use predication instead of branching for some checks, but profile first.

## Sparse Matrix Multiplication

### Algorithm: Two-Pass CSR × CSR

**Input**: Two CSR matrices A (m×k) and B (k×n)  
**Output**: CSR matrix C (m×n)

```
Pass 1: Count non-zeros per output row
  C_indptr[0] = 0
  for each row i in parallel:
    C_indptr[i+1] = count_nnz_in_row(A, B, i)
  C_indptr = prefix_sum(C_indptr)

Pass 2: Compute values
  for each row i in parallel:
    compute_row(A, B, C, i)
```

### Pass 1: Count Non-Zeros

```cuda
template<typename T>
__global__ void count_nnz_kernel(
    const T* __restrict__ A_data,
    const int* __restrict__ A_indices,
    const int* __restrict__ A_indptr,
    const T* __restrict__ B_data,
    const int* __restrict__ B_indices,
    const int* __restrict__ B_indptr,
    int* __restrict__ nnz_counts,
    int m  // Number of rows in A
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    
    // Use hash set to track unique column indices
    // For genealogy graphs: typically 2-20 non-zeros per row
    // Use small fixed-size hash table in shared memory per warp
    
    extern __shared__ int shared_hash[];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int* my_hash = &shared_hash[warp_id * HASH_SIZE];
    
    // Initialize hash table (warp-collaborative)
    for (int i = lane_id; i < HASH_SIZE; i += 32) {
        my_hash[i] = -1;
    }
    __syncwarp();
    
    int count = 0;
    
    // For each non-zero in row of A
    int a_start = A_indptr[row];
    int a_end = A_indptr[row + 1];
    
    for (int a_idx = a_start; a_idx < a_end; a_idx++) {
        T a_val = A_data[a_idx];
        if (a_val == 0) continue;  // Skip explicit zeros
        
        int k = A_indices[a_idx];  // Column in A / row in B
        
        // For each non-zero in row k of B
        int b_start = B_indptr[k];
        int b_end = B_indptr[k + 1];
        
        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
            T b_val = B_data[b_idx];
            if (b_val == 0) continue;
            
            T prod = avos_product(a_val, b_val);
            if (prod == 0) continue;  // Product is zero
            
            int col = B_indices[b_idx];
            
            // Insert into hash table
            bool is_new = hash_insert(my_hash, HASH_SIZE, col);
            if (is_new) count++;
        }
    }
    
    nnz_counts[row] = count;
}
```

### Pass 2: Compute Values

```cuda
template<typename T>
__global__ void compute_values_kernel(
    const T* __restrict__ A_data,
    const int* __restrict__ A_indices,
    const int* __restrict__ A_indptr,
    const T* __restrict__ B_data,
    const int* __restrict__ B_indices,
    const int* __restrict__ B_indptr,
    T* __restrict__ C_data,
    int* __restrict__ C_indices,
    const int* __restrict__ C_indptr,
    int m
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    
    extern __shared__ char shared_mem[];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Hash table for column indices and accumulator for values
    int* col_hash = (int*)&shared_mem[warp_id * 2 * HASH_SIZE * sizeof(int)];
    T* val_accum = (T*)&col_hash[HASH_SIZE];
    
    // Initialize
    for (int i = lane_id; i < HASH_SIZE; i += 32) {
        col_hash[i] = -1;
        val_accum[i] = 0;
    }
    __syncwarp();
    
    // Accumulate products using AVOS sum
    int a_start = A_indptr[row];
    int a_end = A_indptr[row + 1];
    
    for (int a_idx = a_start; a_idx < a_end; a_idx++) {
        T a_val = A_data[a_idx];
        if (a_val == 0) continue;
        
        int k = A_indices[a_idx];
        int b_start = B_indptr[k];
        int b_end = B_indptr[k + 1];
        
        for (int b_idx = b_start; b_idx < b_end; b_idx++) {
            T b_val = B_data[b_idx];
            if (b_val == 0) continue;
            
            T prod = avos_product(a_val, b_val);
            if (prod == 0) continue;
            
            int col = B_indices[b_idx];
            
            // Find or insert column
            int slot = hash_find_or_insert(col_hash, HASH_SIZE, col);
            
            // Accumulate using AVOS sum
            // Need atomic operation to handle potential conflicts
            T old_val = val_accum[slot];
            T new_val = avos_sum(old_val, prod);
            val_accum[slot] = new_val;
        }
    }
    __syncwarp();
    
    // Write results to global memory (sorted by column index)
    int c_start = C_indptr[row];
    int c_end = C_indptr[row + 1];
    int nnz = c_end - c_start;
    
    // Extract non-zeros from hash table and sort
    // (Collaborative within warp)
    if (nnz > 0) {
        extract_and_sort(col_hash, val_accum, HASH_SIZE,
                        &C_indices[c_start], &C_data[c_start], nnz);
    }
}
```

### Helper: Hash Operations

```cuda
__device__ bool hash_insert(int* hash_table, int size, int key) {
    int slot = key % size;
    for (int i = 0; i < size; i++) {
        int probe = (slot + i) % size;
        int old = atomicCAS(&hash_table[probe], -1, key);
        if (old == -1) {
            return true;  // New insertion
        }
        if (old == key) {
            return false;  // Already exists
        }
    }
    // Hash table full - should not happen with proper sizing
    return false;
}

__device__ int hash_find_or_insert(int* hash_table, int size, int key) {
    int slot = key % size;
    for (int i = 0; i < size; i++) {
        int probe = (slot + i) % size;
        int old = atomicCAS(&hash_table[probe], -1, key);
        if (old == -1 || old == key) {
            return probe;
        }
    }
    return -1;  // Error: table full
}
```

## Optimization Strategies

### 1. Shared Memory Usage

**Goal**: Minimize global memory accesses

- **Hash tables**: Store in shared memory (32KB per SM)
- **Row data**: Cache frequently accessed rows
- **Size**: Typical usage ~16KB per block

### 2. Warp-Level Primitives

**Goal**: Reduce synchronization overhead

```cuda
// Warp-level reduction for avos_sum
__device__ T warp_avos_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = avos_sum(val, other);
    }
    return val;
}
```

### 3. Occupancy Tuning

**Target**: 50-75% occupancy

- **Threads per block**: 256 (8 warps)
- **Shared memory**: 16KB per block
- **Registers**: 32-40 per thread
- **Blocks per SM**: 2-4

### 4. Memory Coalescing

**Pattern**: Ensure aligned, contiguous accesses

```cuda
// Good: Coalesced read
int idx = blockIdx.x * blockDim.x + threadIdx.x;
T val = data[idx];  // Adjacent threads read adjacent memory

// Bad: Strided read (avoid)
int idx = threadIdx.x * stride;
T val = data[idx];  // Adjacent threads read far-apart memory
```

## Kernel Launch Configuration

### For Element-wise Operations (avos_sum, avos_product)

```python
def launch_elementwise_kernel(kernel, x, y, out, n):
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel((blocks,), (threads_per_block,), (x, y, out, n))
```

### For Sparse Matrix Multiplication

```python
def launch_sparse_matmul(A, B):
    # Pass 1: Count non-zeros
    threads_per_block = 256
    blocks = (A.shape[0] + threads_per_block - 1) // threads_per_block
    shared_mem_size = (threads_per_block // 32) * HASH_SIZE * 4  # 4 bytes per int
    
    nnz_counts = cupy.zeros(A.shape[0] + 1, dtype=np.int32)
    count_kernel((blocks,), (threads_per_block,), 
                 (A.data, A.indices, A.indptr,
                  B.data, B.indices, B.indptr,
                  nnz_counts, A.shape[0]),
                 shared_mem=shared_mem_size)
    
    # Prefix sum for indptr
    C_indptr = cupy.cumsum(nnz_counts)
    total_nnz = int(C_indptr[-1])
    
    # Pass 2: Compute values
    C_data = cupy.zeros(total_nnz, dtype=A.dtype)
    C_indices = cupy.zeros(total_nnz, dtype=np.int32)
    
    compute_kernel((blocks,), (threads_per_block,),
                   (A.data, A.indices, A.indptr,
                    B.data, B.indices, B.indptr,
                    C_data, C_indices, C_indptr,
                    A.shape[0]),
                   shared_mem=shared_mem_size)
    
    return (C_data, C_indices, C_indptr)
```

## Testing Strategy

1. **Unit tests**: Test each kernel independently
2. **Correctness**: Compare against CPU reference
3. **Edge cases**: Empty matrices, single element, identities
4. **Parity constraints**: Verify all identity behaviors
5. **Performance**: Profile with NVIDIA Nsight

## Next Steps

Read **[03_cupy_integration.md](03_cupy_integration.md)** for Python API and CuPy wrapper design.
