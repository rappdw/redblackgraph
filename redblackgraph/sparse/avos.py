from ._sparsetools import c_avos_sum, c_avos_product

__all__ = ['avos_sum', 'avos_product']

def avos_sum(lhs: int, rhs: int) -> int:
    return c_avos_sum(lhs, rhs)

def avos_product(lhs: int, rhs: int) -> int:
    return c_avos_product(lhs, rhs)