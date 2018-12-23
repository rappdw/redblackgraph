from redblackgraph import rb_matrix
from scipy.sparse import coo_matrix
import numpy as np
from redblackgraph.reference import avos_sum, avos_product
from scipy.sparse.csgraph import depth_first_order

def pass1_w(n, Ap, Aj, Bp, Bj, Cp):
    mask = [-1] * n
    Cp[0] = 0
    nnz = 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i][j] = avos_sum(W[i][j], avos_product(W[i][k], W[k][j]))

def pass1(n_row, n_col, Ap, Aj, Bp, Bj, Cp):
    mask = [-1] * n_col
    Cp[0] = 0
    nnz = 0

    for i in range(n_row):
        row_nnz = 0

        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j+1]):
                k = Bj[kk]
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1

        next_nnz = nnz + row_nnz
        nnz = next_nnz
        Cp[i+1] = nnz

def pass2(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cj, Cx):
    next = [-1] * n_col
    sums = [0] * n_col
    nnz = 0

    for i in range(n_row):
        head = -2
        length = 0

        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            v = Ax[jj]
            for kk in range(Bp[j], Bp[j+1]):
                k = Bj[kk]
                y = Bx[kk]
                sums[k] = avos_sum(sums[k], avos_product(v, y))
                if next[k] == -1:
                    next[k] = head
                    head = k
                    length += 1

        for jj in range(length):
            if sums[head] != 0:
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1
            temp = head
            head = next[head]
            next[temp] = -1
            sums[temp] = 0


    pass

def mul_sparse_matrix(A, B):
    M, k1 = A.shape
    k2, N = B.shape
    indptr = np.empty(M + 1, dtype=A.indptr.dtype)
    pass1(M, N, A.indptr, A.indices, B.indptr, B.indices, indptr)
    indices = np.empty(indptr[-1], dtype=A.indices.dtype)
    data = np.empty(indptr[-1], dtype=A.data.dtype)
    pass2(M, N, A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, indices, data)
    print(indptr)
    print(indices)
    print(data)

def explore_vertex(u, s, l, visited, g):
    if not visited[u]:
        visited[u] = True
        s.append(u)
        for v in range(len(g[0])):
            if g[u] != 0 and u != v:
                s.append((u, v))
    else:
        l.append(u)

def explore_edge(t, s, l, visited, g):
    _, v = t
    if not visited[v]:
        explore_vertex(v, s, l, visited, g)

def rev_topological_ordering(g):
    visited = [False] * len(g)
    s = []
    for u in range(len(g)):
        s.append(u)
    l = []
    while len(s) > 0:
        x = s.pop()
        if isinstance(x, tuple):
            explore_edge(x, s, l, visited, g)
        else:
            explore_vertex(x, s, l, visited, g)
    return l

def topo_sort_util(u, visitied, stack, g):
    visitied[u] = True
    for v in range(len(g[0])):
        if u != v and g[v] and not visitied[v]:
            topo_sort_util(v, visitied, stack, g)
    stack.append(u)

def topo_sort(g):
    visited = [False] * len(g)
    stack = []
    for v in range(len(g)):
        if not visited[v]:
            topo_sort_util(v, visited, stack, g)
    return stack[::-1]

if __name__ == "__main__":
    # super simple, me, dad, mom, parental grandfather
#    matrix = coo_matrix(([-1, 2, 3, -1, 2, 1, -1], ([0, 0, 0, 1, 1, 2, 3], [0, 1, 2, 1, 3, 2, 3])), shape=(4, 4))

    # Arb = [[-1, 2, 3, 0, 0],
    #        [ 0,-1, 0, 2, 0],
    #        [ 0, 0, 1, 0, 0],
    #        [ 0, 0, 0,-1, 0],
    #        [ 2, 0, 0, 0, 1]]
    # order = topo_sort(Arb)
    # print(order)
    # order = rev_topological_ordering(Arb)
    # print(order[::-1])
    matrix = coo_matrix(([-1, 2, 3, -1, 2, 1, -1, 2, 1], ([0, 0, 0, 1, 1, 2, 3, 4, 4], [0, 1, 2, 1, 3, 2, 3, 0, 4])))
    A = rb_matrix(matrix)
    ordering = depth_first_order(A, 4, directed=True, return_predecessors=False)
    print(ordering)
    # print(A[0,2])
    # mul_sparse_matrix(A, A)
    # B = A * A
    # print(B.indptr)
    # print(B.indices)
    # print(B.data)
    # val = A[0, 3]
    # print(val)
    #
    # B = A @ A
    # sorted = B.tocsc().tocsr()
    # # there is a bug on the sparse matrix printing that is exposed by this
    # # test case... to work around it set maxprint to a large number
    # sorted.maxprint = 1000
    # print(f'A = : \n{A}')
    # print(f'A @ A = : \n{sorted}')



    # See MIT Open Courseware 6.006 (lectures 15, 16, ) and 6.046 (lecture 11)
    # Solution for SSSP for a DAG is Topological Sort + Bellman-Ford
    #

