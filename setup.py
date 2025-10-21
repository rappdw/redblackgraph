import setuptools # needed for bdist_wheel
import os
import subprocess
import sys
import sysconfig
import versioneer

from os import path


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('redblackgraph')
    return config

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'redblackgraph'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")

def get_build_ext_override():
    """
    Custom build_ext command to tweak extension building.
    """
    from numpy.distutils.command.build_ext import build_ext as old_build_ext

    class build_ext(old_build_ext):
        def build_extension(self, ext):
            # When compiling with GNU compilers, use a version script to
            # hide symbols during linking.
            if self.__is_using_gnu_linker(ext):
                export_symbols = self.get_export_symbols(ext)
                text = '{global: %s; local: *; };' % (';'.join(export_symbols),)

                script_fn = os.path.join(self.build_temp, 'link-version-{}.map'.format(ext.name))
                with open(script_fn, 'w') as f:
                    f.write(text)
                    # line below fixes gh-8680
                    ext.extra_link_args = [arg for arg in ext.extra_link_args if not "version-script" in arg]
                    ext.extra_link_args.append('-Wl,--version-script=' + script_fn)

            # Allow late configuration
            if hasattr(ext, '_pre_build_hook'):
                ext._pre_build_hook(self, ext)

            old_build_ext.build_extension(self, ext)

        def __is_using_gnu_linker(self, ext):
            if not sys.platform.startswith('linux'):
                return False

            # Fortran compilation with gfortran uses it also for
            # linking. For the C compiler, we detect gcc in a similar
            # way as distutils does it in
            # UnixCCompiler.runtime_library_dir_option
            if ext.language == 'f90':
                is_gcc = (self._f90_compiler.compiler_type in ('gnu', 'gnu95'))
            elif ext.language == 'f77':
                is_gcc = (self._f77_compiler.compiler_type in ('gnu', 'gnu95'))
            else:
                is_gcc = False
                if self.compiler.compiler_type == 'unix':
                    cc = sysconfig.get_config_var("CC")
                    if not cc:
                        cc = ""
                    compiler_name = os.path.basename(cc)
                    is_gcc = "gcc" in compiler_name or "g++" in compiler_name
            return is_gcc and sysconfig.get_config_var('GNULD') == 'yes'

    return build_ext


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

    cmdclass = versioneer.get_cmdclass()

    metadata = dict(
        name="RedBlackGraph",
        version=versioneer.get_version(),
        cmdclass=cmdclass,
        maintainer = "Daniel Rapp",
        maintainer_email = "rappdw@gmail.com",
        description = 'Red Black Graph',
        long_description = long_description,
        long_description_content_type="text/markdown",
        author = "Daniel Rapp",
        download_url = "https://github.com/rappdw/redblackgraph",
        license = 'AGPLv3+',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Version Control :: Git',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        platforms = ["Windows", "Linux", "Mac OS-X"],
        configuration=configuration,
        install_requires=[
            'numpy>=1.26.0,<2.0',
            'scipy>=1.11.0',
            'XlsxWriter',
            'fs-crawler>=0.3.2'
        ],
        extras_require={
            'dev': [
                'wheel>=0.29'
            ],
            'test': [
                'pytest>=3.0',
                'pytest-cov>=2.4',
                'pylint>=1.8.1'
            ],
        },
        setup_requires=[
            'setuptools<60.0',
            'numpy>=1.26.0,<2.0',
            'cython>=3.0'
        ],
        python_requires='>=3.10',
        scripts=[
            'scripts/rbg',
        ]
    )

    # This import is here because it needs to be done before importing setup()
    # from numpy.distutils, but after the MANIFEST removing and sdist import
    # higher up in this file.
    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup

        # Customize extension building
        cmdclass['build_ext'] = get_build_ext_override()

        cwd = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            # Generate Cython sources, unless building from source release
            generate_cython()

        metadata['configuration'] = configuration
    else:
        # Don't import numpy here - non-build actions are required to succeed
        # without Numpy for example when pip is used to install Scipy when
        # Numpy is not yet present in the system.
        metadata['name'] = 'redblackgraph'

    setup(**metadata)
