from redblackgraph import rb_matrix
from scipy.sparse import coo_matrix

if __name__ == "__main__":
    A = rb_matrix(
        coo_matrix(
            (
                [-1, 2, 3, -1, 2, 1, -1],
                (
                    [0, 0, 0, 1, 1, 2, 3],
                    [0, 1, 2, 1, 3, 2, 3]
                )
            ),
            shape=(4, 4)
        )
    )

    B = A @ A
    sorted = B.tocsc().tocsr()
    # there is a bug on the sparse matrix printing that is exposed by this
    # test case... to work around it set maxprint to a large number
    sorted.maxprint = 1000
    print(f'A = : \n{A}')
    print(f'A @ A = : \n{sorted}')

