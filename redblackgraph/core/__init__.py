from __future__ import division, absolute_import, print_function

from .info import __doc__
from numpy.version import version as __version__

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
import os
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    from . import redblackmultiarray
except ImportError as exc:
    msg = """
Importing the multiarray redblackgraph extension module failed.  Most
likely you are trying to import a failed build of redblackgraph.
If you're working with a git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall redblackgraph.

Original error was: %s
""" % (exc,)
    raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os

from . import einsumfunc
from .einsumfunc import *

__all__ = []
__all__ += einsumfunc.__all__
