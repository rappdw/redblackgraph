from __future__ import division, print_function, absolute_import

import os
import sys
import subprocess


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('sparse',parent_package,top_path)

    # config.add_subpackage('linalg')
    # config.add_subpackage('csgraph')

    def get_sparsetools_sources(ext, build_dir):
        # Defer generation of source files
        subprocess.check_call([sys.executable,
                               os.path.join(os.path.dirname(__file__),
                                            'generate_sparsetools.py'),
                               '--no-force'])
        return []

    depends = ['sparsetools_impl.h',
               'rbm.h',
               'rbm_impl.h',
               'sparsetools.h']
    depends = [os.path.join('sparsetools', hdr) for hdr in depends],
    config.add_extension('_sparsetools',
                         define_macros=[('__STDC_FORMAT_MACROS', 1)],
                         depends=depends,
                         include_dirs=['sparsetools'],
                         sources=[os.path.join('sparsetools', 'sparsetools.cxx'),
                                  os.path.join('sparsetools', 'rbm.cxx'),
                                  get_sparsetools_sources]
                         )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
