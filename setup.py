from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-qnns",
    version="1.0.0",
    author="Hybrid-QNNs Research Team",
    author_email="",
    description="A research framework for quantum-classical neural networks and barren plateau analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FAbdullah17/Hybrid-QNNs",
    project_urls={
        "Bug Tracker": "https://github.com/FAbdullah17/Hybrid-QNNs/issues",
        "Documentation": "https://fabdullah17.github.io/Hybrid-QNNs/",
        "Source Code": "https://github.com/FAbdullah17/Hybrid-QNNs",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "ipython>=8.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-qnns=run_all_experiments:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    keywords=[
        "quantum machine learning",
        "quantum computing",
        "neural networks",
        "barren plateaus",
        "quantum entanglement",
        "hybrid models",
        "pennylane",
        "pytorch",
    ],
    zip_safe=False,
)
