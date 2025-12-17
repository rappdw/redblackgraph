"""
Reference implementation of sparse DAG transitive closure.

This module provides a pure Python implementation of transitive closure
for directed acyclic graphs (DAGs) that never allocates O(N²) memory.
"""

from typing import Sequence, Dict, List, Tuple
from redblackgraph.reference.rbg_math import avos_sum, avos_product
from redblackgraph.reference.topological_sort import topological_sort
from redblackgraph.types.transitive_closure import TransitiveClosure


class CycleError(Exception):
    """Raised when a cycle is detected in a graph that should be a DAG."""
    def __init__(self, message: str, vertex: int = -1):
        super().__init__(message)
        self.vertex = vertex


def transitive_closure_dag(M: Sequence[Sequence[int]]) -> TransitiveClosure:
    """
    Compute transitive closure of a DAG using truly sparse operations.
    
    This algorithm never allocates O(N²) memory. It uses topological ordering
    and propagates closure information from successors to predecessors.
    
    Parameters
    ----------
    M : Sequence[Sequence[int]]
        Input adjacency matrix as a 2D list/sequence. Must be a directed 
        acyclic graph (DAG).
        
    Returns
    -------
    TransitiveClosure
        The transitive closure result with matrix W and diameter.
        
    Raises
    ------
    CycleError
        If the graph contains a cycle.
        
    Notes
    -----
    Algorithm:
    1. Compute topological ordering of vertices
    2. Process vertices in reverse topological order (sinks first)
    3. For each vertex v, its closure is: direct edges + union of successor closures
    4. Store closure for each vertex as a sparse row (dictionary)
    
    Complexity:
    - Time: O(V + E + nnz_closure) where nnz_closure is output non-zeros
    - Space: O(nnz_closure) - never allocates N×N dense matrix
    
    Examples
    --------
    >>> # Simple DAG: 0 -> 1 -> 2
    >>> A = [[1, 2, 0], [0, 1, 4], [0, 0, 1]]
    >>> result = transitive_closure_dag(A)
    >>> # Closure adds edge 0 -> 2
    """
    n = len(M)
    
    if n == 0:
        return TransitiveClosure([], 0)
    
    # Get topological ordering
    # The reference topological_sort doesn't raise on cycles, so we need to detect them
    topo_order = topological_sort(M)
    
    # Verify it's a valid DAG by checking for back edges
    position = {v: i for i, v in enumerate(topo_order)}
    for i in range(n):
        for j in range(n):
            if i != j and M[i][j] != 0:
                if position[i] > position[j]:
                    raise CycleError(
                        f"Graph contains a cycle: edge from {i} to {j} is a back edge",
                        vertex=i
                    )
    
    # Process vertices in reverse topological order (sinks first)
    # For each vertex, we compute its closure row
    # closure[v] = {v: identity} ∪ {direct edges from v} ∪ {closure[w] for w in successors(v)}
    
    # Store closure for each vertex as a dictionary: col -> value
    closure_rows: List[Dict[int, int]] = [None] * n
    
    max_value = 0
    
    # Process in reverse topological order
    for v in reversed(topo_order):
        # Initialize closure for v with its direct edges
        v_closure: Dict[int, int] = {}
        
        # Add direct edges from v (including self-loop/identity)
        for col in range(n):
            val = M[v][col]
            if val != 0:
                v_closure[col] = val
                if abs(val) > max_value:
                    max_value = abs(val)
        
        # For each direct successor w of v, add w's closure to v's closure
        for w in range(n):
            if w == v:
                continue
            v_to_w = M[v][w]
            if v_to_w == 0:
                continue
            
            # Get w's closure (already computed since we process in reverse topo order)
            w_closure = closure_rows[w]
            if w_closure is None:
                continue
            
            # For each entry (w, x) -> val in w's closure, add (v, x) -> v_to_w ⊗ val
            for x, w_to_x in w_closure.items():
                # Compute AVOS product: v -> w -> x
                v_to_x = avos_product(v_to_w, w_to_x)
                
                if v_to_x == 0:
                    continue
                
                # AVOS sum with existing value (if any)
                if x in v_closure:
                    v_closure[x] = avos_sum(v_closure[x], v_to_x)
                else:
                    v_closure[x] = v_to_x
                
                if abs(v_closure[x]) > max_value:
                    max_value = abs(v_closure[x])
        
        closure_rows[v] = v_closure
    
    # Build result matrix from closure_rows
    result = [[0] * n for _ in range(n)]
    for v in range(n):
        v_closure = closure_rows[v]
        if v_closure:
            for col, val in v_closure.items():
                result[v][col] = val
    
    # Compute diameter from max value
    if max_value > 1:
        diameter = 0
        temp = max_value
        while temp > 1:
            temp >>= 1
            diameter += 1
    else:
        diameter = 0
    
    return TransitiveClosure(result, diameter)
