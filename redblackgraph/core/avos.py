from ._multiarray import c_einsum_avos, c_avos_sum, c_avos_product

__all__ = ['einsum', 'avos_sum', 'avos_product']

def einsum(*operands, **kwargs):
    avos = kwargs.pop('avos', False)
    if not avos:
        raise ValueError("Avos should be specified in all cases.")
    return c_einsum_avos(*operands, **kwargs)

def avos_sum(lhs: int, rhs: int) -> int:
    return c_avos_sum(lhs, rhs)

def avos_product(lhs: int, rhs: int) -> int:
    return c_avos_product(lhs, rhs)