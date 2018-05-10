from redblackgraph.rb_multiarray import c_einsum_avos, c_avos_sum, c_avos_product

__all__ = ['einsum', 'avos_sum', 'avos_product']

def einsum(*operands, **kwargs):
    avos = kwargs.pop('avos', False)
    if not avos:
        raise ValueError("Avos should be specified in all cases.")
    return c_einsum_avos(*operands, **kwargs)

def avos_sum(*operands):
    return c_avos_sum(*operands)

def avos_product(*operands):
    return c_avos_product(*operands)
