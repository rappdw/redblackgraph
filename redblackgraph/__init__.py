# from .rbm import rb_matrix, Color
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .core import *
from .util import RelationshipFileReader, RedBlackGraphWriter

__all__ = ['__version__']
__all__.extend(core.__all__)
__all__.extend(['RelationshipFileReader', 'RedBlackGraphWriter', 'print'])