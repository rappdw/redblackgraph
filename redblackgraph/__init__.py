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

# Optional file I/O utilities (require fs-crawler and XlsxWriter)
# Install with: pip install redblackgraph[io]
try:
    from .util import RelationshipFileReader, RedBlackGraphWriter, RbgGraphBuilder
    _io_available = True
except ImportError:
    _io_available = False

__all__ = ['__version__']
__all__.extend(core.__all__)
__all__.extend(sparse.__all__)
__all__.extend(['Color'])
__all__.extend(['RED_ONE', 'BLACK_ONE', 'red_one_for_dtype', 'black_one_for_dtype', 
                'is_red_one', 'is_black_one'])
if _io_available:
    __all__.extend(['RelationshipFileReader', 'RedBlackGraphWriter', 'RbgGraphBuilder'])
