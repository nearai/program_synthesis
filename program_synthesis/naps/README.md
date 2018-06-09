# NAPS -- Natural Program Synthesis Dataset
The structure of the NAPS dataset is thoroughly described in the paper: [To be added later]

## Download the dataset
Use the link to download the dataset: https://goo.gl/WaBdbb

## Working with the dataset
[pipelines/](pipelines/) provides examples on how to work with the dataset.
* [pipelines/read_naps.py](pipelines/read_naps.py) demonstrates how to use [pipes/](pipes/) to read the dataset with batching, shuffling, etc;
* [pipelines/validate_naps.py](pipelines/validate_naps.py) uses the execution engine from [uast/uast.py](uast/uast.py) to validate that all solutions in the dataset pass the tests.
Note, though since solutions are human-written some of them might have unintended minor non-determinism caused by e.g. floating point arithmetic, which might result in one or two stray solutions
failing to pass some of the tests;
* [pipelines/pring_naps.py](pipelines/print_naps.py) demonstrates how one might use [uast/uast_pprint.py](uast/uast_pprint.py) to print UAST in nice human-readable format.

## UAST specification
See [uast/README.md](uast/README.md) for the formal specification of UAST.  

## Baseline models
[To be added later]