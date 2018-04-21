import versioneer
import os
from os import path
import subprocess


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
    from numpy.distutils.core import setup

    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    CLASSIFIERS = """\
    Development Status :: 5 - Production/Stable
    Intended Audience :: Science/Research
    Intended Audience :: Developers
    License :: OSI Approved
    Programming Language :: C
    Programming Language :: Python
    Programming Language :: Python :: 2
    Programming Language :: Python :: 2.7
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: Implementation :: CPython
    Topic :: Software Development
    Topic :: Scientific/Engineering
    Operating System :: Microsoft :: Windows
    Operating System :: POSIX
    Operating System :: Unix
    Operating System :: MacOS
    """
    from distutils.command.sdist import sdist


    def check_submodules():
        """ verify that the submodules are checked out and clean
            use `git submodule update --init`; on failure
        """
        if not os.path.exists('.git'):
            return
        with open('.gitmodules') as f:
            for l in f:
                if 'path' in l:
                    p = l.split('=')[-1].strip()
                    if not os.path.exists(p):
                        raise ValueError('Submodule %s missing' % p)

        proc = subprocess.Popen(['git', 'submodule', 'status'],
                                stdout=subprocess.PIPE)
        status, _ = proc.communicate()
        status = status.decode("ascii", "replace")
        for line in status.splitlines():
            if line.startswith('-') or line.startswith('+'):
                raise ValueError('Submodule not clean: %s' % line)


    class sdist_checked(sdist):
        """ check submodules on sdist to prevent incomplete tarballs """

        def run(self):
            check_submodules()
            sdist.run(self)


    metadata = dict(
        version=versioneer.get_version(),
        maintainer = "Daniel Rapp",
        maintainer_email = "rappdw@gmail.com",
        description = 'Red Black Graphs',
        long_description = long_description,
        author = "Daniel Rapp",
        download_url = "https://github.com/rappdw/redblackgraph",
        license = 'MIT',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        # test_suite='nose.collector',
        cmdclass={"sdist": sdist_checked},
        configuration=configuration,
        install_requires=[
            'dataclasses',
            'numpy',
            'XlsxWriter',
        ],
        extras_require={
            'dev': [
                'pytest',
                'pytest-cov',
                'pylint'
            ]
        }
    )

    setup(**metadata)
