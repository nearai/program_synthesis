import collections
import copy

import numpy as np
import torch
import torch.utils.data

from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.tools import shuffle_queue


def queue_to_iter(queue):
    while True:
        item = queue.get()
        if item is None:
            return
        yield item


def make_karel_edit_examples(karel_example, beam_size=None, rng=np.random):
    from program_synthesis.karel.dataset.dataset import KarelEditExample
    goal_atree = refine_env.AnnotatedTree(code=karel_example.code_sequence)
    queue = collections.deque([('DEF', 'run', 'm(', 'm)')])
    closed = set()

    while queue:
        current_code = queue.popleft()
        if current_code in closed:
            continue
        closed.add(current_code)

        current_atree = refine_env.AnnotatedTree(code=current_code)
        actions = refine_env.ComputeAddOps.run(current_atree, goal_atree)
        yield KarelEditExample(
            cur_code=current_code,
            goal_code=karel_example.code_sequence,
            allowed_edits=actions,
            input_tests=karel_example.input_tests,
            tests=karel_example.tests)

        if beam_size is not None and len(actions) > beam_size:
            actions = [
                actions[i]
                for i in rng.choice(
                    len(actions), size=beam_size, replace=False)
            ]
        for action in actions:
            mutation_space = refine_env.MutationActionSpace(
                atree=copy.deepcopy(current_atree))
            mutation_space.apply(action)
            new_code = mutation_space.atree.code
            if new_code not in closed:
                queue.append(new_code)


class SynchronousKarelEditDataLoader(object):
    def __init__(self,
                 karel_dataset,
                 batch_size,
                 collate_fn,
                 beam_size=5,
                 cache=False,
                 shuffle=True,
                 shuffle_queue_size=1000):
        self.karel_dataset = karel_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.beam_size = beam_size
        self.cache = [] if cache else None
        self.cache_complete = False

        if shuffle:
            self.sampler = torch.utils.data.sampler.RandomSampler(karel_dataset)
            self.edit_example_queue_gen = lambda: shuffle_queue.ShuffleQueue(
                shuffle_queue_size, self.karel_edit_example_generator())
        else:
            self.sampler = torch.utils.data.sampler.SequentialSampler(
                karel_dataset)
            self.edit_example_queue_gen = self.karel_edit_example_generator

    def karel_edit_example_generator(self):
        if self.cache:
            for karel_edit_example in self.cache:
                yield karel_edit_example

        for idx in self.sampler:
            karel_example = self.karel_dataset[idx]
            for karel_edit_example in make_karel_edit_examples(
                    karel_example, self.beam_size):
                if self.cache is not None:
                    self.cache.append(karel_edit_example)
                yield karel_edit_example

    def __iter__(self):
        batch = []
        for karel_edit_example in self.edit_example_queue_gen():
            batch.append(karel_edit_example)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if len(batch) > 0:
            yield self.collate_fn(batch)


class KarelEditDataLoader(object):
    # Main process starts N workers.
    #
    # In each worker:
    # - Grab a KarelExample
    # - Generate M
    '''KarelExample -> KarelEditExample workers:
    - consumes a KarelExample from train.pkl
    - generates M KarelEditExamples.
    - Adds the KarelEditExamples to a queue.

    Shuffling worker:
    - consumes a KarelEditExample from queue
    - adds to ShuffleQueue

    KarelEditExample -> batch workers:
    - Consumes from ShuffleQueue
    - Runs the batch processor
    - Adds to the final queue

    This process:
    - Gets a batch from the final queue and then provides it
    '''

    def __init__(self, karel_dataset):
        torch.utils.data.sampler.RandomSampler(karel_dataset)

    @staticmethod
    def _shuffle_worker(in_queue, out_queue, buffer_size):
        # in_queue: multiproessing.Queue, contains (pickled) KarelEditExamples.
        # out_queue: multiprocessing.Queue
        shuffler = shuffle_queue.ShuffleQueue(buffer_size,
                                              queue_to_iter(in_queue))
        while True:
            try:
                out_queue.put(next(shuffler))
            except StopIteration:
                out_quue.put(None)
                return

    @staticmethod
    def _batch_worker(in_queue, out_queue, collator):
        pass
