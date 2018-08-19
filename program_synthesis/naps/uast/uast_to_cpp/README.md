# UAST to C++ converter

* Converts UAST to C++
* Compiles the program into a shared library;
* Runs the shared library against the tests from the dataset.

The main purpose is to be able to sandbox inferred UAST programs. We have noticed that programs inferred by our model
contain large number of outliers that either take extremely large time or memory to execute.

Currently, as of 06/26/2018, the converter has 99% success rate converting UAST into a compilable C++ program and 95% chance that the
program passes all the tests.

### Example
See `compile_run_naps.py` for an example on how to use it.

### Requirements
Should run on Linux and macOS with the most recent clang. Tested on:

**Ubuntu 16.04**, clang version 6.0.1, ld 2.26.1

**macOS 10.13.5**, Apple LLVM version 9.1.0 (clang-902.0.39.2)
