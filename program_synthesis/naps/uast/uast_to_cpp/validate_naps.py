"""
Iterates over the NAPS dataset, converts UAST to C++, compiles and executes it validating that C++ passes all tests.
"""
from itertools import chain

from program_synthesis.naps.pipelines.read_naps import read_naps_dataset
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.pipe import Pipe
from program_synthesis.naps.uast.uast_to_cpp.cpp_executor import execute_program


class FilterPartial(Pipe):
    def __init__(self, is_partial_key):
        self.is_partial_key = is_partial_key

    def __iter__(self):
        for d in self.input:
            if d[self.is_partial_key]:
                continue
            yield d
        return


if __name__ == "__main__":
    trainA, trainB, test = read_naps_dataset()
    trainB = Compose([
        trainB,
        FilterPartial(is_partial_key="is_partial")
    ])
    with trainA, trainB, test:
        for d in chain(trainA, trainB, test):
            execute_program(d['code_tree'], d['tests'])
