from __future__ import division, print_function, absolute_import

import os


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('core', parent_package, top_path)

    src_dir = os.path.join('src', 'multiarray')
    config.add_extension('_multiarray',
                         [
                             os.path.join(src_dir, 'rbg_math.h.src'),
                             os.path.join(src_dir, 'rbg_math.c.src'),
                             os.path.join(src_dir, 'redblack.c.src'),
                             os.path.join(src_dir, 'relational_composition.c.src'),
                             os.path.join(src_dir, 'warshall.c.src')
                         ],
                         include_dirs=[src_dir])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
