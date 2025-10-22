# from .rbm import rb_matrix, Color
try:
    from ._version import __version__
except ImportError:
    # Version file not yet generated (e.g., in development without build)
    __version__ = "0.0.0.dev0"

from .core import *
from .sparse import *
from .util import RelationshipFileReader, RedBlackGraphWriter, RbgGraphBuilder
from redblackgraph.types.color import Color

__all__ = ['__version__']
__all__.extend(core.__all__)
__all__.extend(sparse.__all__)
__all__.extend(['RelationshipFileReader', 'RedBlackGraphWriter', 'Color'])
