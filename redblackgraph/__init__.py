# from .rbm import rb_matrix, Color
try:
    from ._version import __version__
except ImportError:
    # Version file not yet generated (e.g., in development without build)
    __version__ = "0.0.0.dev0"

from .core import *
from .sparse import *
from redblackgraph.types.color import Color

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
if _io_available:
    __all__.extend(['RelationshipFileReader', 'RedBlackGraphWriter', 'RbgGraphBuilder'])
