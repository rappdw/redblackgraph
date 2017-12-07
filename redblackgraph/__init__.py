# from .rbm import rb_matrix, Color
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import core
from .core import *

__all__ = ['__version__']
__all__.extend(core.__all__)
