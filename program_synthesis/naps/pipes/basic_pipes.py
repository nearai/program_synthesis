import json
import random
import numpy as np


from .pipe import Pipe


class JsonLoader(Pipe):
    def __iter__(self):
        for l in self.input:
            try:
                yield json.loads(l)
            except ValueError:
                pass
        return

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return json.loads(self.input[item])


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

    def __len__(self):
        return len(self.input)


class DropKeys(Pipe):
    def __init__(self, keys_to_exclude=None):
        self.keys_to_exclude = keys_to_exclude or set()

    def __iter__(self):
        for el in self.input:
            yield {k: v for k, v in el.items() if k not in self.keys_to_exclude}
        return

    def __getitem__(self, item):
        return {k: v for k, v in self.input[item].items() if k not in self.keys_to_exclude}

    def __len__(self):
        return len(self.input)


class RandomAccessFile(Pipe):
    """
    Provides random access to a file.
    """
    def __init__(self, filename):
        self.filename = filename

    def enter(self):
        self.offsets = []
        with open(self.filename) as f:
            while True:
                offset = f.tell()
                if f.readline():
                    self.offsets.append(offset)
                else:
                    break
        self.f = open(self.filename)

    def exit(self):
        self.offsets.clear()
        self.f.close()

    def __getitem__(self, item):
        self.f.seek(self.offsets[item])
        if item < len(self) - 1:
            return self.f.readline().rstrip('\n')  # Remove the newline character.
        else:
            return self.f.readline()

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        self.f.seek(0)
        return iter(self.f)


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

    def __len__(self):
        num_batches = len(self.input) // self.batch_size
        last_batch = len(self.input) % self.batch_size
        if last_batch and not self.drop_last:
            num_batches += 1
        return num_batches


class SortBatchByLen(Pipe):
    def __init__(self, key):
        self.key = key

    def __iter__(self):
        for b in self.input:
            yield sorted(b, key=lambda x: len(x[self.key]), reverse=True)
        return


class EndlessShuffleCycle(Pipe):
    def __iter__(self):
        while True:
            indices = list(range(len(self.input)))
            random.shuffle(indices)
            for idx in indices:
                yield self.input[idx]


class WeightedMerge(Pipe):
    def __init__(self, input_pipes, p=None):
        """
        Merges the input form several pipes. Iterates between them until one exits.
        :param input_pipes: (list of Pipe objects): pipes to merge;
        :param p: list of floats, weights of the pipes, can be not normalized.
        """
        self.input_pipes = input_pipes
        self.p = [float(el)/sum(p) for el in p]

    def __enter__(self):
        for pipe in self.input_pipes:
            pipe.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for pipe in reversed(self.input_pipes):
            pipe.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        input_pipes = [iter(pipe) for pipe in self.input_pipes]
        pipeline_indices = list(range(len(input_pipes)))
        while True:
            idx = np.random.choice(pipeline_indices, p=self.p)
            pipe = input_pipes[idx]
            try:
                yield next(pipe)
            except StopIteration:
                return

    def __len__(self):
        return sum(len(p) for p in self.input_pipes)


class LimitOutput(Pipe):
    def __init__(self, max_output_num):
        """
        Limits the output of the pipeline.
        :param max_output_num: int, maximum number of elements to output.
        """
        self.max_output_num = max_output_num

    def enter(self):
        self.counter = 0

    def __iter__(self):
        for d in self.input:
            if self.counter < self.max_output_num:
                self.counter += 1
                yield d
            else:
                break
        return


class Identity(Pipe):
    def __iter__(self):
        for d in self.input:
            yield d
        return
