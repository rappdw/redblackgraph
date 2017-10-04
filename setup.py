import versioneer
from setuptools import setup, find_packages, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

sparse_tools = Extension('_sparsetools',
                         sources=['redblackgraph/sparsetools/sparsetools.cxx', 'redblackgraph/sparsetools/rbm.cxx'],
                         include_dirs=get_numpy_include_dirs())

setup(
    name='redblackgraph',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
        'numpy==1.13.3',
        'scipy==1.0.0rc1',
        'jupyter==1.0.0',
        'jupyter_contrib_nbextensions==0.3.1',
        'jupyter_nbextensions_configurator==0.2.7',
        'matplotlib==2.1.0rc1',
        'networkx==2.0',
        'nxpd==0.2.0',
        'sympy==1.1.1'
    ],

    extras_require={
        'dev': [
            'wheel==0.30.0'
        ],
        'test': [
            'pytest==3.2.2',
            'pytest-cov==2.5.1'
        ],
    },

    package_data={
    },

    ext_modules=[sparse_tools]
)
