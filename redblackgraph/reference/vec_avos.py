from functools import reduce
from typing import Sequence
from redblackgraph.reference import avos_product, avos_sum

def vec_avos(u: Sequence[int], v: Sequence[int]) -> int:
    '''Given two vectors, compute the avos product.'''
    return reduce( avos_sum, [avos_product(a, b) for a, b in zip(u, v)])
