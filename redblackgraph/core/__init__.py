from __future__ import division, absolute_import, print_function

__all__ = []

try:
    from .. import rb_multiarray
    from ..rb_multiarray import warshall, vertex_relational_composition, vertex_relational_composition2, edge_relational_composition
except ImportError as exc:
    msg = """
Importing the multiarray redblackgraph extension module failed.  Most
likely you are trying to import a failed build of redblackgraph.
If you're working with a git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall redblackgraph.

Original error was: %s
""" % (exc,)
    raise ImportError(msg)

from . import avos_einsumfunc
from .avos_einsumfunc import *
from . import redblack
from .redblack import *

__all__ += avos_einsumfunc.__all__
__all__ += redblack.__all__
__all__ += ['warshall', 'vertex_relational_composition', 'edge_relational_composition']
