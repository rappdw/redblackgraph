from setuptools import setup, find_packages, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs, get_pkg_info

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

sparse_tools = Extension('_sparsetools',
                         sources=['redblackgraph/sparsetools/sparsetools.cxx', 'redblackgraph/sparsetools/rbm.cxx'],
                         include_dirs=get_numpy_include_dirs())

setup(
    name='redblackgraph',
    use_scm_version={
        'write_to': 'redblackgraph/about.py'
    },
    setup_requires=['setuptools_scm'],
    description='Linear algebra for a specialized ajacency matrix',
    long_description=long_description,
    url='',

    author='Daniel Rapp',
    author_email='rappdw@gmail.com',

    license='MIT',
    keywords='library',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        # Set this topic to what works for you
        'Topic :: Python :: Library',
        # Pick your license as you wish (should match "license" above)
        'License :: MIT',
        'Programming Language :: Python :: 3.5',
    ],

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.19.0'
    ],

    extras_require={
        'dev': [
            'wheel==0.29.0'
        ],
        'test': [
            'pytest==3.0.7'
        ],
    },

    package_data={
    },

    ext_modules=[sparse_tools]
)
