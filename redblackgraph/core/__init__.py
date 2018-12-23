from __future__ import division, absolute_import, print_function

__all__ = []

try:
    from ._multiarray import warshall, vertex_relational_composition, edge_relational_composition
except ImportError as exc:
    msg = """
Importing the multiarray redblackgraph extension module failed.  Most
likely you are trying to import a failed build of redblackgraph.
If you're working with a git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall redblackgraph.

Original error was: %s
""" % (exc,)
    raise ImportError(msg)

from .avos import *
from .redblack import *

__all__ += avos.__all__
__all__ += redblack.__all__
