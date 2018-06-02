import argparse
import collections
import multiprocessing
import os

import numpy as np
import six

if six.PY2:
    import cPickle as pickle
else:
    import pickle
import gzip
import random
import json
import time

import torch.utils.data
from program_synthesis.algolisp.dataset import data


Schema = collections.namedtuple("Schema", ["args", "return_type"])


def _basepath(path):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), path)


def build_vocabs(data_, min_freq=50):
    """Builds separate vocabs for text and code."""
    word_freqs = collections.defaultdict(int)
    code_freqs = collections.defaultdict(int)
    for example in data_:
        for word in example.text:
            word_freqs[word] += 1
        for word in example.code_sequence:
            code_freqs[word] += 1
    return data.get_vocab(word_freqs, min_freq), data.get_vocab(code_freqs, min_freq)


def build_vocab(data_, min_freq=50):
    """Builds single vocab."""
    freqs = collections.defaultdict(int)
    def update_freqs(words):
        for word in words:
            freqs[word] += 1
    for example in data_:
        update_freqs(example.text)
        update_freqs(example.code_sequence)
        for column in example.schema.args.items():
            update_freqs(column)
    return data.get_vocab(freqs, min_freq)


class CodeFunc(object):

    def __init__(
            self, name, schema,
            code_tree, code_sequence):
        self.name = name
        self.schema = schema
        self.code_tree = code_tree
        self.code_sequence = code_sequence

    def to_dict(self):
        return {
            'name': self.name,
            'return_type': self.schema.return_type,
            'args': [(name, type_) for name, type_ in self.schema.args.items()],
            'code_tree': self.code_tree,
        }

    @classmethod
    def from_dict(cls, d):
        # TODO: Don't pass None for code_sequence
        return cls(d['name'],
                   Schema(d['args'], d['return_type']), d['short_tree'], None)


class CodeExample(object):

    def __init__(
            self, text, schema, input_tests,
            code_tree, code_sequence, funcs=[], tests=[],
            candidate_code_sequence=None,
            task_types=[], tags=[], language='lisp'):
        self.text = text
        self.schema = schema
        self.input_tests = input_tests
        self.code_tree = code_tree
        self.code_sequence = code_sequence
        self.funcs = funcs
        self.tests = tests
        # Add candidate_code_tree in the future
        self.candidate_code_sequence = candidate_code_sequence
        self.task_types = task_types
        self.tags = tags
        self.language = language

    def to_dict(self):
        return {
            'text': self.text,
            'return_type': self.schema.return_type,
            'args': [(arg, type_) for arg, type_ in self.schema.args.items()],
            'code_sequence': self.code_sequence,
            'code_tree': self.code_tree,
            'funcs': [f.to_dict() for f in self.funcs],
            'tests': self.input_tests + self.tests,
            'tags': self.tags,
            'nodes': self.task_types,
            'language': self.language
        }

    @classmethod
    def from_dict(cls, d, input_test_ratio=0.7):
        input_test_count = int(len(d['tests']) * input_test_ratio)
        return cls(
            d['text'],
            Schema(d['args'], d['return_type']),
            d['tests'][:input_test_count],
            d['short_tree'],
            d['code_sequence'], [CodeFunc.from_dict(f) for f in d['funcs']],
            d['tests'][input_test_count:],
            task_types=d['nodes'],
            tags=d['tags'])


class NearDataset(object):

    def _line_to_code_example(self, line):
        try:
            line = json.loads(line)
        except ValueError:
            return None
        args = line['args']
        if not isinstance(args, dict):
            args = collections.OrderedDict(args)
        return_type = line.get('return_type', None)
        language = line['language'] if 'language' in line else 'lisp'
        if 'text' in line and line['text']:
            text = line['text']
            if not isinstance(text, list):
                try:
                    text = data.tokenize_text_line(text)
                except Exception as e:
                    print("Exception while tokenizing %s" % text)
                    print(e)
                    return None
        elif 'statement' in line and line['statement']:
            try:
                text = data.tokenize_text_line(line['statement'])
            except Exception as e:
                print("Exception while tokenizing %s" % line['statement'])
                print(e)
                return None
        else:
            text = []
        funcs = [
            CodeFunc(
                name=func['name'],
                schema=Schema(func['args'], func['return_type']),
                code_tree=func['short_tree'],
                code_sequence=data.flatten_code(func['short_tree']))
            for func in line['funcs']
        ] if 'funcs' in line else []

        code_tree = code_sequence = None
        if 'uast' in line and line['uast']:
            code_tree = line['uast']
            language = 'uast'
            if 'code_sequence' in line and line['code_sequence']:
                code_sequence = line['code_sequence']
            else:
                code_sequence = data.flatten_code(code_tree, language)
        elif 'short_tree' in line and line['short_tree']:
            code_tree = line['short_tree']
            code_sequence = data.flatten_code(code_tree, language)
        elif 'code_tree' in line and line['code_tree']:
            code_tree = line['code_tree']
            if isinstance(code_tree, dict):
                language = 'uast'
            if 'code_sequence' in line and line['code_sequence']:
                code_sequence = line['code_sequence']
            else:
                try:
                    code_sequence = data.flatten_code(code_tree, language)
                except Exception as e:
                    print(e)
                    print(line['tags'])
                    print(code_tree)
                    raise
        elif 'code_sequence' in line:
            code_sequence = line['code_sequence']
        if not isinstance(code_sequence, list):
            code_sequence = data.tokenize_code_line(line['code_sequence'])

        if self.strip_brackets:
            code_sequence = [x for x in code_sequence if x not in ('(', ')', '[')]
        if self.filter_code_length > 0 and len(code_sequence) > self.filter_code_length:
            return None
        if self.max_code_length > 0 and code_sequence is not None:
            code_sequence = code_sequence[:self.max_code_length]

        if self.skip_empty_code and not code_tree and not code_sequence:
            print("Found no code in record: %s" % line)
            return None

        if len(line['tests']) > 1:
            if len(line['tests']) < 4:
                split_idx = 1
            elif len(line['tests']) < 5:
                split_idx = 2
            else:
                split_idx = 3
            input_tests, tests = line['tests'][:split_idx], line['tests'][split_idx:]
        else:
            input_tests, tests = [], line['tests']

        example = CodeExample(
            text=text,
            schema=Schema(args, return_type),
            code_sequence=code_sequence,
            code_tree=code_tree,
            funcs=funcs,
            input_tests=input_tests,
            tests=tests,
            task_types=line['nodes'] if 'nodes' in line else [],
            tags=line['tags'] if 'tags' in line else [],
            language=language
        )

        if self.pseudocode_match:
            example.code_tree = base._prepare_code_tree(example.code_tree, True, False)
            example.match = example.text
            example.text = []
        return example

    def build_vocabs(self, min_freq=50, ignore_constants=False):
        return build_vocabs(self.data, min_freq)

    def build_vocab(self, min_freq=50):
        return build_vocab(self.data, min_freq)

    class TorchNearDataset(torch.utils.data.Dataset):
        def __init__(self, _len, _get_item):
            self._len = _len
            self._get_item = _get_item

        def __len__(self):
            return self._len()

        def __getitem__(self, idx):
            return self._get_item(idx)

    def augment_example(self, example):
        return example

    def _get_item(self, idx):
        if self.load_in_ram:
            return self.augment_example(self.cache[idx])
        else:
            offset = self.offsets[idx]
            self.f.seek(offset)
            line = self.f.readline()
            example = self._line_to_code_example(line)
            # This assert should always pass because we have filtered invalid examples when constructing offsets.
            assert example is not None
            return self.augment_example(example)

    def _len(self):
        return len(self.cache) if self.load_in_ram else len(self.offsets)

    def __init__(self, filename, args, shuffle=False, 
                 max_code_length=0, filter_code_length=0,
                 skip_empty_code=True, 
                 pseudocode=False,  strip_brackets=False,
                 variable_shuffle=False,
                 pseudocode_match=False,
                 is_training=True):
        batch_size = args.batch_size
        max_size = getattr(args, 'dataset_max_size', 0)
        self.load_in_ram = getattr(args, 'dataset_load_in_ram', False)
        self.max_code_length = max_code_length
        self.filter_code_length = filter_code_length
        self.skip_empty_code = skip_empty_code
        self.pseudocode = pseudocode
        self.strip_brackets = strip_brackets
        self.variable_shuffle = variable_shuffle
        self.pseudocode_match = pseudocode_match
        self.f = open(filename)

        if self.load_in_ram:
            self.cache = []
        else:
            self.offsets = []

        with open(filename) as tmp_f:
            if self.pseudocode_match:
                templates = json.loads(tmp_f.readline())
            while True:
                # We should not use f as an iterator, because tell() will not work correctly
                # due to the read-ahead buffering. https://stackoverflow.com/a/19731163
                offset = tmp_f.tell()
                line = tmp_f.readline()
                if not line:
                    break
                example = self._line_to_code_example(line)
                if example is not None:
                    if self.load_in_ram:
                        self.cache.append(example)
                    else:
                        self.offsets.append(offset)
                    if max_size > 0 and self._len() >= max_size:
                        break

        print("Loaded %s, total records: %d" % (filename, self._len()))
        if self.pseudocode_match:
            templates = [base.CodeTemplate.from_dict(dct) for dct in templates]
            self.matcher = base.Matcher(templates, [])
            print("Loaded matcher with %d templates." % (len(templates)))

        # Create dataset and DataLoader using offset.
        self.data = NearDataset.TorchNearDataset(self._len, self._get_item)

        if args.dataset_bucketing:
            self.iterator = BucketIterator(
                dataset=self.data,
                batch_size=batch_size,
                sort_key=lambda ex: interleave_keys(len(ex.text), len(ex.code_sequence)),
                sort_within_batch_key=lambda ex: len(ex.text),
                shuffle=shuffle,
                sort_within_batch=True,  # Sort in desc order using sort_key,
                marcobucket_size=args.dataset_macrobucket_size,
                sort_all=not is_training
            )
        else:
            def collate_fn(batch):
                # Sort sequences by their length so that we don't have to worry about packing/unpacking sequences.
                return sorted(batch, key=lambda ex: len(ex.text), reverse=True)
            self.iterator = torch.utils.data.DataLoader(self.data,
                                                        batch_size=batch_size,
                                                        # Note, unlike previous implementation of NearDataset, this
                                                        # shuffle will reshuffle the data at each epoch.
                                                        shuffle=shuffle,
                                                        num_workers=0,
                                                        collate_fn=collate_fn,
                                                        drop_last=False)

    def __iter__(self):
        return self.iterator.__iter__()

    def __next__(self):
        return self.iterator.__iter__().__next__()

    def __len__(self):
        return self.iterator.__len__()


def get_metagen_dataset(args):
    args.code_vocab = args.word_vocab = _basepath('data/generated/word.vocab')
    train_data = NearDataset(
        _basepath('data/generated/metaset5.train.jsonl'),
        args, shuffle=True, max_code_length=getattr(args, 'dataset_max_code_length', 0),
        filter_code_length=getattr(args, 'dataset_filter_code_length', 0),
        is_training=True)
    if not os.path.exists(args.word_vocab):
        data.save_vocab(args.word_vocab, train_data.build_vocab(min_freq=args.vocab_min_freq))
    dev_data = NearDataset(
        _basepath('data/generated/metaset5.dev.jsonl'),
        args, shuffle=False,
        is_training=False)
    return train_data, dev_data


def get_metagen_eval_dataset(args):
    args.code_vocab = args.word_vocab = _basepath('data/generated/word.vocab')
    return NearDataset(
        _basepath('data/generated/metaset5.dev.jsonl'),
        args, shuffle=True,
        is_training=False)


def get_dataset(args):
    if args.dataset == 'metagen':
        return get_metagen_dataset(args)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)


def get_eval_dataset(args):
    if args.dataset == 'handcrafted':
        return get_metagen_handcrafted_dataset(args)
    elif args.dataset == 'metagen':
        return get_metagen_eval_dataset(args)
    else:
        raise ValueError("Unknown dataset %s" % args.dataset)


def dataset_split(args, dataset, filenames, proportions, dedup_non_first=False):
    def _renormalize(lst):
        total = sum(lst)
        return [float(x) / total for x in lst]
    datastats = [stats.DatasetStats(args) for _ in filenames]
    files = [open(filename, 'w') for filename in filenames]
    real_proportions = [x for x in proportions]
    candidates = list(range(len(proportions)))
    expected_size = [len(dataset.data) * p for p in proportions]
    for example in dataset.data:
        fidx = -1
        for i, s in enumerate(datastats):
            if (example.code_sequence and str(example.code_sequence) in s.code_map) or (example.text and str(example.text) in s.text_map):
                fidx = i
        # If this is not first file and we dedup the rest, skip duplicate solutions for this text.
        if fidx > 0 and dedup_non_first:
            continue
        if fidx == -1:
            fidx = np.random.choice(candidates, p=proportions)
        datastats[fidx].update(example)
        files[fidx].write(json.dumps(example.to_dict()) + "\n")
        if datastats[fidx].stats['total'] >= expected_size[fidx] and fidx in candidates:
            idx = candidates.index(fidx)
            candidates.pop(idx)
            proportions.pop(idx)
            proportions = _renormalize(proportions)

    for f in files:
        f.close()
    for i, ds in enumerate(datastats):
        print("=== %.2f%% ===" % real_proportions[i])
        ds.display()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-test-split', action='store_true', default=False)
    parser.add_argument('--train-dev-split', action='store_true', default=False)
    parser.add_argument('--original', type=str, default=None)
    parser.add_argument('--show_tags', action='store_true', default=False)
    parsed_args = parser.parse_args()

    dataset_args = collections.namedtuple('Args', ['batch_size'])(batch_size=1)

    base_name = '.'.join(parsed_args.original.split('.')[:-1])

    if parsed_args.train_test_split:
        d = NearDataset(parsed_args.original, dataset_args, shuffle=False, skip_empty_code=False)
        print("Loaded dataset from %s" % parsed_args.original)
        dataset_split(
            parsed_args, d,
            [base_name + '.train.jsonl',
             base_name + '.dev.jsonl',
             base_name + '.test.jsonl'],
            [0.8, 0.1, 0.1])

    if parsed_args.train_dev_split:
        start = time.time()
        d = NearDataset(parsed_args.original, dataset_args, shuffle=True, skip_empty_code=False)
        print("Loaded dataset from %s in %ss" % (parsed_args.original, time.time() - start))
        dataset_split(
            parsed_args, d,
            [base_name + '.train.jsonl',
             base_name + '.dev.jsonl'],
            [0.9, 0.1], dedup_non_first=True
        )
