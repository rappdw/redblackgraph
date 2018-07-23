import versioneer
import sys

from os import path


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('redblackgraph',
                           parent_package,
                           top_path)
    config.add_extension('rb_multiarray',
                         [
                             'redblackgraph/core/src/multiarray/rbg_math.h.src',
                             'redblackgraph/core/src/multiarray/rbg_math.c.src',
                             'redblackgraph/core/src/multiarray/redblack.c.src',
                             'redblackgraph/core/src/multiarray/relational_composition.c.src',
                             'redblackgraph/core/src/multiarray/warshall.c.src'
                         ],
                         include_dirs=['redblackgraph/core/src/multiarray'])

    # config.add_extension('rb_sparsetools',
    #                      [
    #                         'redblackgraph/core/src/sparsetools/sparsetools.cxx',
    #                         'redblackgraph/core/src/sparsetools/rbm.cxx'
    #                      ],
    #                      include_dirs=['redblackgraph/core/src/sparsetools'])

    return config


if __name__ == "__main__":

    args = sys.argv[1:]

    run_build = True
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            run_build = False

    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    metadata = dict(
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        maintainer = "Daniel Rapp",
        maintainer_email = "rappdw@gmail.com",
        description = 'Red Black Graphs',
        long_description = long_description,
        author = "Daniel Rapp",
        download_url = "https://github.com/rappdw/redblackgraph",
        license = 'MIT',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Version Control :: Git',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        configuration=configuration,
        install_requires=[
            'dataclasses;python_version<"3.7"',
            'numpy>=0.14.0',
            'XlsxWriter',
        ],
        extras_require={
            'dev': [
                'pytest',
                'pytest-cov',
                'pylint'
            ]
        },
        setup_requires=[
            'numpy>=0.14.0',
        ]
    )

    # This import is here because it needs to be done before importing setup()
    # from numpy.distutils, but after the MANIFEST removing and sdist import
    # higher up in this file.
    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup

        # # Customize extension building
        # cmdclass['build_ext'] = get_build_ext_override()
        #
        # cwd = os.path.abspath(os.path.dirname(__file__))
        # if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        #     # Generate Cython sources, unless building from source release
        #     generate_cython()

        metadata['configuration'] = configuration
    else:
        # Don't import numpy here - non-build actions are required to succeed
        # without Numpy for example when pip is used to install Scipy when
        # Numpy is not yet present in the system.

        # Version number is added to metadata inside configuration() if build
        # is run.
        #metadata['version'] = get_version_info()[0]
        metadata['name'] = 'redblackgraph'
        pass

    setup(**metadata)
