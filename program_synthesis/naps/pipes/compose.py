"""
Example of basic usage:

batching_pipeline = Compose([
            open("mydataset.jsonl"),
            JsonLoader(),
            Cache(),
            Batcher(batch_size=100)])

preparation_pipeline = Compose([
            batching_pipeline,
            ExtractFeatures(),
            TensorCreator()])
"""

import copy

from .pipe import Pipe, CallablePipe


class Compose(Pipe):
    def __init__(self, pipes):
        """
        Composes several pipes together. Will wrap callable objects into CallablePipe.
        :param pipes: (list of Pipe or callable objects): list of pipes to compose.
        """
        self._pipes = [CallablePipe(p) if not isinstance(p, Pipe) and callable(p) else p for p in pipes]
        # Connect inputs of the pipes.
        for curr, prev in zip(self._pipes[1:], self._pipes[:-1]):
            curr.input = prev

    def __iter__(self):
        return iter(self._pipes[-1])

    def enter(self):
        self._pipes[-1].__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pipes[-1].__exit__(exc_type, exc_val, exc_tb)

    def __copy__(self):
        return Compose([copy.copy(p) for p in self._pipes])

    def __getitem__(self, item):
        return self._pipes[-1][item]

    def __len__(self):
        return len(self._pipes[-1])
