# from .rbm import rb_matrix, Color
try:
    from ._version import __version__
except ImportError:
    # Version file not yet generated (e.g., in development without build)
    __version__ = "0.0.0.dev0"

from .core import *
from .sparse import *
from redblackgraph.types.color import Color
from .constants import (
    RED_ONE, BLACK_ONE,
    red_one_for_dtype, black_one_for_dtype,
    is_red_one, is_black_one
)

__all__ = ['__version__']
__all__.extend(core.__all__)
__all__.extend(sparse.__all__)
__all__.extend(['Color'])
__all__.extend(['RED_ONE', 'BLACK_ONE', 'red_one_for_dtype', 'black_one_for_dtype',
                'is_red_one', 'is_black_one'])


def __getattr__(name):
    """Lazy import for optional I/O utilities (require fs-crawler and XlsxWriter)."""
    if name in ('RelationshipFileReader', 'RedBlackGraphWriter'):
        from .util import relationship_file_io
        return getattr(relationship_file_io, name)
    if name == 'RbgGraphBuilder':
        from .util.graph_builder import RbgGraphBuilder
        return RbgGraphBuilder
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
