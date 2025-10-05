"""
Setup script for BloomWatch.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bloomwatch",
    version="1.0.0",
    author="BloomWatch Team",
    author_email="contact@bloomwatch.ai",
    description="A reproducible pipeline for satellite vegetation bloom detection and prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/bloomwatch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bloomwatch-fetch-gee=src.data_fetch.fetch_gee:main",
            "bloomwatch-fetch-appeears=src.data_fetch.fetch_appeears:main",
            "bloomwatch-preprocess=src.preprocessing.preprocess_ndvi:main",
            "bloomwatch-label=src.labeling.label_from_change_point:main",
            "bloomwatch-features=src.features.features:main",
            "bloomwatch-train=src.models.train_model:main",
            "bloomwatch-evaluate=src.models.evaluate:main",
            "bloomwatch-api=src.api.app:main",
            "bloomwatch-demo=src.demo.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md"],
    },
)
