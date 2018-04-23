from functools import reduce
from typing import Sequence
from redblackgraph.reference import avos_product, avos_sum


def mat_avos(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
    '''Given two matrices, compute the "avos" product.'''
    return [[reduce( avos_sum, [avos_product(a, b) for a, b in zip(A_row, B_col)]) for B_col in zip(*B)] for A_row in A]
