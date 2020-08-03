#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re

from setuptools import setup, find_packages
# from setuptools.extension import Extension
# from Cython.Build import cythonize
# import numpy as np

classifiers = """\
    Development Status :: 4 - Beta
    Operating System :: OS Independent
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
"""


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read('wiggin', '__init__.py'),
        re.MULTILINE).group(1)
    return version


def get_long_description():
    return _read('README.md')


install_requires = [
    'polychrom',
    'looplib'
]


packages = find_packages()
setup(
    name='wiggin',
    author='Anton Goloborodko',
    author_email='goloborodko.anton@gmail.com',
    version=get_version(),
    license='MIT',
    description='An experimental high-level modular framework for building polymer simulations.',
    long_description=get_long_description(),
    keywords=['polymer simulations',],
    url='https://github.com/golobor/wiggin',
    packages=find_packages(),
    # ext_modules = cythonize(extensions),
    zip_safe=False,
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    # include_dirs=[np.get_include()],

    install_requires=install_requires,

)
