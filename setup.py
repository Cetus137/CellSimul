#!/usr/bin/env python3
"""
CellSimul - A Python package for generating synthetic cell membrane masks
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt if it exists
def get_requirements():
    """Get the requirements from requirements.txt"""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Default requirements
        requirements = [
            "numpy>=1.19.0",
            "pandas>=1.3.0", 
            "scikit-image>=0.18.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "h5py>=3.0.0",
            "tifffile>=2021.1.0",
            "matplotlib>=3.3.0"
        ]
    return requirements

setup(
    name="cellsimul",
    version="0.1.0",
    author="CellSimul Team",
    author_email="",
    description="A Python package for generating synthetic cell membrane masks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cetus137/CellSimul",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cellsimul=src.synthetic_cell_membrane_masks_2d:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
