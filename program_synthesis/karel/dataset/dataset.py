import argparse
import collections
import gzip
import json
import os
import random
import struct
import sys
import six
import time

if six.PY2:
    import cPickle as pickle
else:
    import pickle

import numpy as np
import torch.utils.data

from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import executor
from program_synthesis.karel.dataset import stats
from program_synthesis.karel.dataset.mutation import KarelExampleMutator
from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.dataset import edit_data_loader
from program_synthesis.karel.tools import indexed_file
from program_synthesis.karel.tools import batch_creators

Schema = collections.namedtuple("Schema", ["args", "return_type"])


def relpath(path):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), path)


class KarelExample(object):
    __slots__ = (
         'idx',
        'guid',
        'code_sequence',
        'input_tests',
        'tests',
        'text',
        'ref_example',
        'resplit')
    schema = Schema(None, None)
    code_tree = []
    _empty_trace = executor.KarelTrace([], [], [])

    def __init__(self, idx, guid, code_sequence, input_tests, tests,
            ref_example=None):
        self.idx = idx
        self.guid = guid
        self.code_sequence = code_sequence
        self.input_tests = input_tests
        self.tests = tests
        self.text = code_sequence
        self.ref_example = ref_example
        self.resplit = False

    @classmethod
    def from_dict(cls, d):
        all_examples = []
        for example in d['examples']:
            ex = {
                'input': sorted(list(int(x) for x in example['in'])),
                'output': sorted(list(int(x) for x in example['out']))
            }
            if 'trace' in example:
                ex['trace'] = example['trace']
            elif 'trace_grids' in example:
                ex['trace'] = executor.KarelTrace(
                        grids=example['trace_grids'],
                        events=[])
            if 'traces' in example:
                ex['traces'] = example['traces']
            all_examples.append(ex)
        assert len(all_examples) == 6
        ref_dict = d.get('ref')
        if ref_dict:
            ref_example = KarelExample.from_dict(ref_dict)
        else:
            ref_example = None
        return cls(d.get('id', None), d['guid'], d['code'], all_examples[:5], all_examples[5:],
                ref_example)

    def to_dict(self):
        return {
            'id': self.idx,
            'guid': self.guid,
            'examples': [{
                'in': example['input'],
                'out': example['output'],
                'trace': example.get('trace', self._empty_trace),
                'trace_grids': example.get('trace', self._empty_trace).grids,
                'traces': example.get('traces', []),
            } for example in self.input_tests + self.tests],
            'code': self.code_sequence,
            'ref': self.ref_example.to_dict() if self.ref_example else None
        }

    def resplit_examples(self, batch_creator):
        assert not self.resplit
        if batch_creator.is_shuffled:
            random.shuffle(self.input_tests + self.tests)
        self.input_tests, self.tests = batch_creator.from_examples(
                self.input_tests + self.tests)
        self.resplit = True
        if self.ref_example:
            self.ref_example.resplit_examples(batch_creator)


class KarelEditExample(object):

    __slots__ = ('cur_code', 'goal_code', 'allowed_edits', 'input_tests',
                 'tests')

    def __init__(self,
            cur_code,
            goal_code,
            allowed_edits,
            input_tests,
            tests):
        self.cur_code = cur_code
        self.goal_code = goal_code
        self.allowed_edits = allowed_edits
        self.input_tests = input_tests
        self.tests = tests

    def resplit_examples(self, batch_creator):
        assert not self.resplit
        if batch_creator.is_shuffled:
            random.shuffle(self.input_tests + self.tests)
        self.input_tests, self.tests = batch_creator.from_examples(
                self.input_tests + self.tests)
        self.resplit = True

class BucketizedSampler(object):

    def __init__(self, dataset, buckets, bucket_key, adaptive_size=None):
        self.dataset = dataset
        self.buckets = buckets
        self.adaptive_size = adaptive_size
        self.bucket_ids = {k: [] for k in self.buckets}
        for idx, example in enumerate(self.dataset.data):
            key = bucket_key(example)
            self.bucket_ids[key].append(idx)
        print("Buckets: " + ", ".join(['%s: %s' % (key, len(self.bucket_ids[key])) for key in buckets]))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.dataset.shuffle:
            for key in self.bucket_ids:
                random.shuffle(self.bucket_ids[key])
        self._last_i = {key: 0 for key in self.bucket_ids}
        return self

    def next(self):
        non_empty_keys = [key for key in self.bucket_ids if self._last_i[key] < len(self.bucket_ids[key])]
        if not non_empty_keys:
            raise StopIteration
        res = []
        key = random.choice(non_empty_keys)
        while self._last_i[key] < len(self.bucket_ids[key]) and len(res) < self.dataset.batch_size:
            res.append(self.dataset.data[self.bucket_ids[key][self._last_i[key]]])
            self._last_i[key] += 1
            if self.adaptive_size and self.adaptive_size(res):
                break
        return res


class Dataset(object):

    def __init__(self, batch_size, data, shuffle=False):
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle

    def __iter__(self):
        self._index = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self._index)
        self._last_i = 0
        return self

    def __len__(self):
        return (len(self.data) - 1) // self.batch_size + 1

    def next(self):
        if self._last_i == len(self.data):
            raise StopIteration
        res = []
        while self._last_i < len(self.data) and len(res) < self.batch_size:
            res.append(self.data[self._index[self._last_i]])
            self._last_i += 1
        return res

    def build_vocab(self, min_freq=50):
        freqs = collections.defaultdict(int)
        def update_freqs(words):
            for word in words:
                freqs[word] += 1
        for example in self.data:
            update_freqs(example.text)
            update_freqs(example.code_sequence)
            for column in example.schema.args.items():
                update_freqs(column)
        return data.get_vocab(freqs, min_freq)

    def save(self, filename):
        with open(filename, 'w') as f:
            for example in self.data:
                f.write(json.dumps(example.to_dict()) + "\n")


class DynamicDataset(object):
    SHARD_SIZE = 100

    def __init__(self, batch_size, capacity=None, min_items=None, path=None):
        self.items = collections.deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_items = min_items
        if self.min_items and self.capacity:
            assert self.capacity >= self.min_items

        self.path = path
        if self.path is not None:
            self.shard_sizes = collections.deque()
            self.shard_items_count = 0
            if os.path.exists(self.path):
                entries =  os.listdir(self.path)
                entries.sort(key=int)
                print('Loading from {}...'.format(self.path))
                for entry in entries:
                    with gzip.open(os.path.join(self.path, entry)) as f:
                        shard = pickle.load(f)
                        self.shard_items_count += len(shard)
                        self.shard_sizes.append(len(shard))
                        self.items.extend(shard)
                print('Done.')

                if entries:
                    self.earliest_shard = int(entries[0])
                    self.next_shard = int(entries[-1]) + 1
                else:
                    self.earliest_shard = 0
                    self.next_shard = 0
                self.candidate_shard = []
            else:
                os.mkdir(self.path)
                self.earliest_shard = 0
                self.next_shard =  0
                self.candidate_shard = []

    def next(self):
        if len(self.items) <= self.batch_size:
            return list(self.items)

        return random.sample(self.items, self.batch_size)

    def add(self, item):
        self.items.append(item)
        if self.path is None:
            return

        self.candidate_shard.append(item)
        if len(self.candidate_shard) == DynamicDataset.SHARD_SIZE:
            with gzip.open(os.path.join(self.path, str(self.next_shard)), 'w') as f:
                pickle.dump(self.candidate_shard, f, pickle.HIGHEST_PROTOCOL)
            self.shard_items_count += DynamicDataset.SHARD_SIZE
            self.shard_sizes.append(DynamicDataset.SHARD_SIZE)
            self.next_shard += 1
            self.candidate_shard = []

            while self.shard_items_count - self.shard_sizes[0] >= self.capacity:
                self.shard_sizes.popleft()
                os.unlink(os.path.join(self.path, str(self.earliest_shard)))
                self.earliest_shard += 1

    def __len__(self):
        return (len(self.items) - 1) // self.batch_size + 1

    def is_ready(self):
        if self.min_items:
            return len(self.items) > self.min_items
        return bool(self.items)


class KarelTorchDataset(torch.utils.data.Dataset):

    def __init__(self, filename, mutator=lambda x: x):
        self.filename = filename
        self.mutator = mutator

        self.file = None
        self.index = indexed_file.read_index(self.filename + '.index')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = open(self.filename, 'rb')
        self.file.seek(self.index[idx])

        if six.PY2:
            return self.mutator(KarelExample.from_dict(pickle.load(self.file)))
        else:
            return self.mutator(KarelExample.from_dict(pickle.load(self.file, encoding='latin-1')))


class KarelDataset(object):

    def __init__(self, filename, batch_size, mutator=lambda x: x):
        self.filename = filename
        self.batch_size = batch_size
        self.file = open(self.filename, 'rb')
        self.mutator = mutator

    def __iter__(self):
        self.file.seek(0)
        return self

    def next(self):
        res = []
        try:
            while len(res) < self.batch_size:
                res.append(
                    self.mutator(
                        KarelExample.from_dict(pickle.load(self.file))))
        except EOFError:
            pass
        if not res:
            raise StopIteration
        return res

    def build_vocab(self):
        tokens = collections.defaultdict(int)
        self.file.seek(0)
        while True:
            try:
                example = pickle.load(self.file)
            except EOFError:
                break
            for token in example['code']:
                tokens[token] += 1
        return data.get_vocab(tokens, 1)


def get_karel_dataset(args, model, section, for_eval, num_async_workers,
                      shuffle, alt_path, batch_creator_spec):
    suffix = args.dataset[5:]

    if args.karel_mutate_ref:
        mutation_dist = [float(x) for x in args.karel_mutate_n_dist.split(',')]
        mutator = KarelExampleMutator(mutation_dist, rng_fixed=for_eval,
                add_trace=args.karel_trace_enc != 'none')
    else:
        mutator = lambda x: x

    if alt_path:
        path = try_paths(
                relpath('../../' + alt_path),
                alt_path)
    else:
        path = relpath('../../data/karel/{}{}.pkl'.format(section, suffix))

    karel_dataset = KarelTorchDataset(path, mutator)
    batch_creator = eval(batch_creator_spec, vars(batch_creators))
    collate_fn = batch_creators.collate_wrapper(
        model.batch_processor(for_eval=for_eval), batch_creator)

    if args.model_type == 'karel-edit':
        if args.load_sync:
            return edit_data_loader.SynchronousKarelEditDataLoader(
                    karel_dataset,
                    args.batch_size,
                    collate_fn,
                   beam_size=None if for_eval else args.karel_edit_data_beam,
                   shuffle=not for_eval,
                   shuffle_queue_size=10000)
        else:
            raise ValueError('Only --load-sync supported for now')

    return torch.utils.data.DataLoader(
        karel_dataset,
        args.batch_size,
        collate_fn=collate_fn,
        num_workers=0 if args.load_sync else num_async_workers,
        pin_memory=False,
        shuffle=args.karel_train_shuf and not for_eval)


def get_karel_train_dataset(args, model, for_eval=False):
    return get_karel_dataset(
        args,
        model,
        'train',
        for_eval,
        num_async_workers=4,
        shuffle=args.karel_train_shuf and not for_eval,
        alt_path=args.train_data_path,
        batch_creator_spec=args.batch_create_train)


def get_karel_eval_dataset(args, model):
    return get_karel_dataset(
        args,
        model,
        'val',
        for_eval=True,
        num_async_workers=2,
        shuffle=False,
        alt_path=args.eval_data_path,
        batch_creator_spec=args.batch_create_eval)


def get_karel_eval_final_dataset(args, model):
    return get_karel_dataset(
        args,
        model,
        'test',
        for_eval=True,
        num_async_workers=2,
        shuffle=False,
        alt_path=args.eval_data_path,
        batch_creator_spec=args.batch_create_eval)


def set_vocab(args):
    if args.dataset.startswith('karel'):
        args.word_vocab = relpath('../../data/karel/word.vocab')
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)


def get_train_dataset(args, model, for_eval):
    if args.dataset.startswith('karel'):
        return get_karel_train_dataset(args, model, for_eval)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)


def get_eval_dataset(args, model):
    if args.dataset.startswith('karel'):
        return get_karel_eval_dataset(args, model)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)


def get_eval_final_dataset(args, model):
    if args.dataset.startswith('karel'):
        return get_karel_eval_final_dataset(args, model)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)
