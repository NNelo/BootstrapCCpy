#!/usr/bin/env python
from distutils.core import setup

description = 'CC library for python3'
keywords = ['clustering']

setup(
    name='BootstrapCCpy',
    version='0.2.2',
    description=description,
    author='FB, NN, EF, PP',
    package_data={'': ['Readme.md']},
    license="MIT",
    install_requires=[
        'numpy',
        'joblib',
        'kneed',
        'matplotlib',
        'scipy'
    ],
    url='https://github.com/NNelo/BootstrapCCpy',
    py_modules=['BootstrapCCpy'],
    keywords=keywords
)
