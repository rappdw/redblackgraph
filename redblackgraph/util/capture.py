import redblackgraph as rb
from typing import Sequence, Union

def _capture_list(A):
    vertices = range(len(A))
    data = '[\n'
    for i in vertices:
        if i > 0:
            data += ',\n'
        data += '  ['
        for j in vertices:
            if j > 0:
                data += ','
            if A[i][j] != -1 and A[i][j] < 10:
                data += ' '
            data += str(A[i][j])
        data += ']'
    data += '\n]'
    return data

def _capture_rb_matrix(A):
    N = A.shape[0]
    A1 = [[0] * N for _ in range(N)]
    for i, j in zip(*A.nonzero()):
        A1[i][j] = A[i, j]
    return _capture_list(A1)

def capture(A: Union[Sequence[Sequence[int]], rb.rb_matrix]):
    if isinstance(A, rb.rb_matrix):
        return _capture_rb_matrix(A)
    else:
        return _capture_list(A)

