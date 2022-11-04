#!/usr/bin/env python
from setuptools import setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name='vmci_segmentation',
          version="1.0",
          maintainer='Lisa Gutzeit',
          maintainer_email='lisa.gutzeit@uni-bremen.de',
          description='Python implementation of velocity-based multiple change-point inference (vMCI)',
          long_description=long_description,
          long_description_content_type="text/markdown",
          license='BSD-3-Clause',
          packages=['vmci_segmentation'],
          install_requires=["numpy", "scipy", "matplotlib"],
          extras_require={
              "doc": ["numpydoc", "sphinx", "sphinx-gallery", "sphinxcontrib-bibtex"],
              "test": ["pytest"]
          }
    )