###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from setuptools import find_packages, setup

setup(
    name="primuspipe",
    version="0.1.0",
    description="A flexible and scalable pipeline parallelism scheduling framework",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
    ],
    extras_require={
        "example": [
            "torchvision",
        ],
        "simulation": [
            "pyyaml",
            "matplotlib",
            "plotext",
        ],
    },
)
