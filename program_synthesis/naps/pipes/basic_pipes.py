import json
import numpy as np
import random


from .pipe import Pipe


class JsonLoader(Pipe):
    def __iter__(self):
        for l in self.input:
            try:
                yield json.loads(l)
            except ValueError:
                pass
        return


class JsonDumper(Pipe):
    def __iter__(self):
        return (json.dumps(d) for d in self.input)


class Cache(Pipe):
    """
    Caches the input before providing the sequential or the random access.
    """
    def enter(self):
        self.cache = None

    def exit(self):
        if self.cache:
            self.cache.clear()

    def _run_caching(self):
        if self.cache:
            return
        self.cache = []
        for d in self.input:
            self.cache.append(d)

    def __iter__(self):
        self._run_caching()
        return iter(self.cache)

    def __getitem__(self, item):
        self._run_caching()
        return self.cache[item]

    def __len__(self):
        self._run_caching()
        return len(self.cache)


class KeepKeys(Pipe):
    def __init__(self, keys_to_include=None):
        self.keys_to_include = keys_to_include or set()

    def __iter__(self):
        for el in self.input:
            yield {k: v for k, v in el.items() if k in self.keys_to_include}
        return

    def __getitem__(self, item):
        return {k: v for k, v in self.input[item].items() if k in self.keys_to_include}


class DropKeys(Pipe):
    def __init__(self, keys_to_exclude=None):
        self.keys_to_exclude = keys_to_exclude or set()

    def __iter__(self):
        for el in self.input:
            yield {k: v for k, v in el.items() if k not in self.keys_to_exclude}
        return

    def __getitem__(self, item):
        return {k: v for k, v in self.input[item].items() if k not in self.keys_to_exclude}


class Cycle(Pipe):
    def __init__(self, shuffle=True, times=None):
        """
        Endlessly cycles the input pipe. Expects the pipeline to provide random access.
        :param shuffle: If True will access input in random order;
        :param times: (int): if not None, will cycle over the input collection given number of times.
        """
        self.shuffle = shuffle
        self.times = times

    def __iter__(self):
        iteration = 0
        while self.times is None or iteration < self.times:
            iteration += 1
            indices = list(range(len(self.input)))
            if self.shuffle:
                random.shuffle(indices)
            for idx in indices:
                yield self.input[idx]
        return


class WeightedMerge(Pipe):
    def __init__(self, input_pipes, weights):
        """
        Randomly reads from the pipes according to their weight.
        :param input_pipes: (list of Pipe objects): pipes to merge;
        :param weights:
        """
        self.input_pipes = input_pipes
        self.weights = weights

    def normalize(self, weights):
        return np.array(weights, dtype=np.float) / sum(weights)

    def __enter__(self):
        for pipe in self.input_pipes:
            pipe.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for pipe in reversed(self.input_pipes):
            pipe.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        input_pipes = [iter(p) for p in self.input_pipes]
        pipeline_indices = list(range(len(input_pipes)))
        weights = self.normalize(self.weights)
        while True:
            idx = np.random.choice(pipeline_indices, p=weights)
            pipe = input_pipes[idx]
            try:
                yield next(pipe)
            except StopIteration:
                self.pipeline_indices.pop(idx)
                self.weights.pop(idx)
                if not self.input_pipes:
                    # All input pipes have ended.
                    return
                weights = self.normalize(weights)


class Batch(Pipe):
    def __init__(self, batch_size, drop_last=False):
        """
        Batches input pipe.
        :param batch_size: (int): size of the batch;
        :param drop_last: (bool): whether to skip last incomplete batch.
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for el in self.input:
            batch.append(el)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
        return
