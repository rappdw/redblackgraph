from redblackgraph.rb_multiarray import c_einsum_avos

__all__ = ['einsum']

def einsum(*operands, **kwargs):
    avos = kwargs.pop('avos', False)
    if not avos:
        raise ValueError("Avos should be specified in all cases.")
    return c_einsum_avos(*operands, **kwargs)
