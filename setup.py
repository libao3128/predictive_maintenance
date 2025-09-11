#!/usr/bin/env python3
"""
Setup script for inverter-predictive-maintenance package.
"""

from setuptools import setup, find_packages
import os
import sys

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Predictive maintenance system for solar plant inverters using deep learning"

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle version constraints properly
                    requirements.append(line)
    return requirements

# Read development requirements
def read_dev_requirements():
    requirements = []
    if os.path.exists("requirements-dev.txt"):
        with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-r"):
                    requirements.append(line)
    return requirements

setup(
    name="inverter-predictive-maintenance",
    version="1.0.0",
    author="UCLA MEng Capstone Team",
    author_email="leo900527@gmail.com",
    maintainer="UCLA MEng Capstone Team",
    maintainer_email="leo900527@gmail.com",
    description="Predictive maintenance system for solar plant inverters using deep learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/libao3128/predictive_maintenance",
    download_url="https://github.com/libao3128/predictive_maintenance/archive/v1.0.0.tar.gz",
    packages=find_packages(),
    platforms=["any"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements() or [
            "pytest>=6.0.0,<7.0.0",
            "pytest-cov>=2.12.0,<3.0.0",
            "black>=21.0.0,<22.0.0",
            "flake8>=3.9.0,<4.0.0",
            "isort>=5.9.0,<6.0.0",
            "mypy>=0.910,<1.0.0",
            "jupyter>=1.0.0,<2.0.0",
            "notebook>=6.0.0,<7.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0,<5.0.0",
            "sphinx-rtd-theme>=0.5.0,<1.0.0",
            "myst-parser>=0.15.0,<1.0.0",
        ],
        "azure": [
            "azure-storage-blob>=12.0.0,<13.0.0",
            "azure-identity>=1.6.0,<2.0.0",
            "azure-ml>=1.0.0,<2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "inverter-predictive-maintenance=inverter_predictive_maintenance.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "inverter_predictive_maintenance": [
            "config/*.json",
            "*.md",
            "*.yml",
            "*.yaml",
        ],
    },
    data_files=[
        ("config", ["config/dataset_parameters.json", "config/model_parameters.json"]),
        ("", ["requirements.txt", "requirements-dev.txt", "environment.yml"]),
    ],
    keywords=[
        "predictive maintenance",
        "solar energy",
        "inverter",
        "machine learning",
        "deep learning",
        "time series",
        "CNN-LSTM",
        "failure prediction",
        "renewable energy",
    ],
    project_urls={
        "Bug Reports": "https://github.com/libao3128/predictive_maintenance/issues",
        "Source": "https://github.com/libao3128/predictive_maintenance",
        "Documentation": "https://github.com/libao3128/predictive_maintenance/blob/main/README.md",
    },
)
