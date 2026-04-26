#!/usr/bin/env python
"""Setup script for vit-chest-xray package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
requirements = []
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="vit-chest-xray",
    version="1.0.0",
    author="Ehtasham Arif",
    author_email="ehtasham.arif@example.com",
    description="Vision Transformer with Clinical Attention Consistency for Chest X-Ray Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ehtasham-Arif01/vit-chest-xray",
    packages=find_packages(exclude=["tests", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "vis": [
            "plotly>=5.14.0",
            "dash>=2.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vit-train=src.train:main",
            "vit-evaluate=src.evaluate:main",
            "vit-visualize=src.visualize:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)