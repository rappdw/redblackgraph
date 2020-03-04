from typing import Sequence

def permute(A:Sequence[Sequence[int]], p:Sequence[int]) -> Sequence[Sequence[int]]:
    '''Permutes an input matrix based on the vertex ordering specified.

    Equivalent to P * A * P-1 (where P is a permutation of the identity matrix specified by p)
    '''
    n = len(A)
    vertices = range(n)
    B = [[0 for _ in vertices] for _ in vertices]
    for old_idx, new_idx in enumerate(p):
        for j in vertices:
            B[old_idx][j] = A[new_idx][p[j]]
    return B
