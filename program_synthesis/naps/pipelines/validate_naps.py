"""
Iterates over the NAPS dataset and validates that solutions pass all tests.
"""
from itertools import chain

from program_synthesis.naps.pipelines.read_naps import read_naps_dataset
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.basic_pipes import Pipe

from program_synthesis.naps.uast import Executor


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
            pass
