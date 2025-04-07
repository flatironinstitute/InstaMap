#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from setuptools import find_namespace_packages, setup

import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="instamap.nn._instamapops",
        sources=[
            "src/cpp/library.cpp",
            "src/cpp/rotation_mse.cpp",
            "src/cpp/rotation_mse_cuda.cu",
        ],
        include_dirs=[os.path.join(os.path.abspath(os.path.dirname(__file__)), "third_party/libcudacxx/include")],
        extra_compile_args=["-g"]),
]

setup(
    name='InstaMap',
    version='1.0.0',
    license='MIT',
    description='InstaMap: instant-NGP for cryo-EM density maps',
    author='Geoffrey Woollard, Wenda Zhou, Erik H. Thiede, Chen Lin, Nikolaus Grigorieff, Pilar Cossio, Khanh Dao Duc, Sonya M. Hanson',
    author_email='geoffwoollard@gmail.com, , ehthiede@gmail.com, , , , , shanson@flatironinstitute.org',
    url='',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'': ['*.yaml', '*.csv', '*.json']},
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.8',
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
)
