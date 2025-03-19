# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup, find_packages

setup(
    name='AMD-gpt-fast',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    description='A simple, fast, pure PyTorch Llama inference engine with multimodal support',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AMD-AIG-AIMA/AMD-gpt-fast',
)
