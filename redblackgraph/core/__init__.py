from __future__ import division, absolute_import, print_function

try:
    from .. import rb_multiarray
except ImportError as exc:
    msg = """
Importing the multiarray redblackgraph extension module failed.  Most
likely you are trying to import a failed build of redblackgraph.
If you're working with a git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall redblackgraph.

Original error was: %s
""" % (exc,)
    raise ImportError(msg)

from . import einsumfunc
from .einsumfunc import *
from . import redblack
from .redblack import *

__all__ = []
__all__ += einsumfunc.__all__
__all__ += redblack.__all__
