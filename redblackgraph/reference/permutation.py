from typing import Sequence

def permute(A:Sequence[Sequence[int]], p:Sequence[int], assume_upper_triangular: bool = False) -> Sequence[Sequence[int]]:
    '''Permutes an input matrix based on the vertex ordering specified.

    Equivalent to P * A * P-1 (where P is a permutation of the identity matrix specified by p)
    '''
    n = len(A)
    vertices = range(n)
    B = [[0 for _ in vertices] for _ in vertices]
    for i in vertices:
        start = i if assume_upper_triangular else 0
        for j in range(start, n):
            B[i][j] = A[p[i]][p[j]]
    return B
