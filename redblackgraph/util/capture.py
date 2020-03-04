from typing import Sequence

def capture(A: Sequence[Sequence[int]]):
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