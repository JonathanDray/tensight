from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tensight",
    version="0.1.0",
    author="Jonathan Dray",
    author_email="",
    description="See through your PyTorch models - Advanced analysis and diagnostics toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonathanDray/tensight",
    packages=find_packages(where="tensight"),
    package_dir={"": "tensight"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    keywords="pytorch, deep-learning, neural-networks, analysis, diagnostics, loss-landscape, gradient-noise",
    project_urls={
        "Bug Reports": "https://github.com/JonathanDray/tensight/issues",
        "Source": "https://github.com/JonathanDray/tensight",
        "Documentation": "https://github.com/JonathanDray/tensight#readme",
    },
)

