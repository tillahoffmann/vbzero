from setuptools import find_packages, setup


setup(
    name="vbzero",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "matplotlib",
            "myst-nb",
            "sphinx",
            "sphinx_rtd_theme",
        ]
    }
)
