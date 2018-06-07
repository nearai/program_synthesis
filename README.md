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
```
pip install program-synthesis
```

## Development installation

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
