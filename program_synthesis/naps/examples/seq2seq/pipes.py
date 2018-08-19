import random
import numpy as np
import multiprocessing

from program_synthesis.naps.uast import uast_to_lisp, lisp_to_uast
from program_synthesis.naps.uast.lisp_to_uast import lisp_to_uast
from program_synthesis.naps.pipes.pipe import Pipe


class SplitTests(Pipe):
    def __init__(self, tests_key, input_tests_key, eval_tests_key):
        self.tests_key = tests_key
        self.input_tests_key = input_tests_key
        self.eval_tests_key = eval_tests_key

    def __iter__(self):
        for d in self.input:
            tests = d.get(self.tests_key, [])
            num_tests = len(tests)
            if num_tests == 1:
                split_idx = 0
            elif num_tests < 4:
                split_idx = 1
            elif num_tests < 5:
                split_idx = 2
            else:
                split_idx = 3
            yield {**d, **{self.input_tests_key: tests[:split_idx],
                           self.eval_tests_key: tests[split_idx:]}}
        return


class SelectPseudocode(Pipe):
    def __iter__(self):
        for d in self.input:
            yield {**d, **{"text": random.choice(d["texts"])}}
        return


class FilterCodeLength(Pipe):
    def __init__(self, dataset_filter_code_length):
        self.dataset_filter_code_length = dataset_filter_code_length

    def __iter__(self):
        for d in self.input:
            if len(d['code_sequence']) < self.dataset_filter_code_length:
                yield d
        return

class SortBatchByLen(Pipe):
    def __init__(self, key):
        self.key = key

    def __iter__(self):
        for b in self.input:
            yield sorted(b, key=lambda x: len(x[self.key]), reverse=True)
        return


class ShuffleVariables(Pipe):
    def __init__(self, code_tree_key, code_sequence_key, text_key):
        self.code_tree_key = code_tree_key
        self.code_sequence_key = code_sequence_key
        self.text_key = text_key

    def make_remap(self, names, prefix, upto):
        cur = list(names[prefix].values())
        values = ['%s%d' % (prefix, i) for i in range(upto)]
        random.shuffle(values)
        return dict(zip(cur, values))

    def __iter__(self):
        for d in self.input:
            names = {'struct': {}, 'func': {}, 'var': {}}
            uast_to_lisp.remap_uast(d[self.code_tree_key], names)
            remap = self.make_remap(names, 'var', 35)
            remap.update(self.make_remap(names, 'func', 7))
            remap.update(self.make_remap(names, 'struct', 2))
            new_text = [remap.get(word, word) for word in d[self.text_key]]
            new_code_sequence = [remap.get(token, token) for token in d[self.code_sequence_key]]
            new_code_tree = lisp_to_uast(new_code_sequence)
            yield {**d, **{self.text_key: new_text,
                           self.code_sequence_key: new_code_sequence,
                           self.code_tree_key: new_code_tree}}
        return


class SkipPartial(Pipe):
    def __init__(self, is_partial_key):
        self.is_partial_key = is_partial_key

    def __iter__(self):
        for d in self.input:
            if not d.get(self.is_partial_key, False):
                yield d
        return


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


class EndlessShuffleCycle(Pipe):
    def __iter__(self):
        while True:
            indices = list(range(len(self.input)))
            random.shuffle(indices)
            for idx in indices:
                yield self.input[idx]


class Identity(Pipe):
    def __iter__(self):
        for d in self.input:
            yield d
        return

def worker(compose, queue, buffer_closed, workers_done, worker_idx):
    queue.cancel_join_thread()
    with compose:
        for el in compose:
            if buffer_closed.value == 1:
                break
            queue.put(el)
    workers_done[worker_idx] = 1


class Buffer(Pipe):
    def __init__(self, compose, max_buffer_size=10, num_workers=1):
        self.compose = compose
        self.max_buffer_size = max_buffer_size
        self.num_workers = num_workers

    def enter(self):
        self.queue = multiprocessing.Queue(self.max_buffer_size)
        self.buffer_closed = multiprocessing.Value('B', 0)
        self.workers_done = multiprocessing.Array('B', [0]*self.num_workers)
        self.workers = [multiprocessing.Process(target=worker, args=(self.compose, self.queue, self.buffer_closed,
                                                                     self.workers_done, idx))
                        for idx in range(self.num_workers)]
        for w in self.workers:
            w.start()

    def __iter__(self):
        while True:
            if self.all_workers_done:
                break
            yield self.queue.get()
        return

    @property
    def all_workers_done(self):
        return all(w == 1 for w in self.workers_done[:])

    def exit(self):
        # Send the signal to the workers to stop.
        self.buffer_closed.value = 1
        # Deplete the queue in case there were blocked workers before we sent the signal.
        while not self.all_workers_done:
            self.queue.get()
        # Close the queue.
        self.queue.close()
        self.workers.clear()