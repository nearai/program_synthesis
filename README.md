<img src="logo.jpg" width=30% align="right" />

[![PyPi version](https://pypip.in/v/program-synthesis/badge.png)](https://pypi.org/project/program-synthesis/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1299382.svg)](https://doi.org/10.5281/zenodo.1299382)
# Program Synthesis
NEAR Program Synthesis provides a set of models, tools, and datasets for program synthesis tasks.

This repository will make it easier for the community to compare and reuse program synthesis algorithms across different
datasets.

## Prerequisites
Python 3 (>=3.5) is required to run the code. We also recommend using
[virtualenv](https://virtualenv.pypa.io/en/stable/) for isolated Python environments and
[pip](https://pypi.org/project/pip/) for package management. Note, to create a Python 3 environment you need to run:

```
virtualenv .env --python=python3
source .env/bin/activate
```

The code also assumes that [PyTorch](https://pytorch.org/) is already installed.

## Installation

For development installation you need to clone the repository:
```
git clone https://github.com/nearai/program_synthesis.git
cd program_synthesis
```

Install program-synthesis in editable mode:
```
pip install -e .
```

## Models and datasets
- [AlgoLisp](program_synthesis/algolisp)
- [Karel](program_synthesis/karel)
- [NAPS](program_synthesis/naps)

To cite this repository in publications:

    @misc{illia_polosukhin_2018_1299382,
      author       = {Illia Polosukhin and
                      Maksym Zavershynskyi and
                      Richard Shin},
      title        = {nearai/program_synthesis: v0.1.2},
      month        = jun,
      year         = 2018,
      doi          = {10.5281/zenodo.1299382},
      url          = {https://doi.org/10.5281/zenodo.1299382}
    }
