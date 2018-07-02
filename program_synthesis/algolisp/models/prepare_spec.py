import collections
import itertools
import operator

import numpy as np

import torch
from torch.autograd import Variable


def lengths(lsts):
    return [len(lst) for lst in lsts]


def lists_to_numpy(lsts, stoi, default_value):
    max_length = max(lengths(lsts))
    data = np.full((len(lsts), max_length), default_value, dtype=np.int64)
    for i, lst in enumerate(lsts):
        for j, element in enumerate(lst):
            data[i, j] = stoi(element)
    return data


def numpy_to_tensor(arr, cuda, volatile):
    t = torch.LongTensor(arr)
    if cuda:
        t = t.cuda()
    return Variable(t, volatile=volatile)


def numpy_to_float_tensor(arr, cuda, volatile):
    t = torch.FloatTensor(arr)
    if cuda:
        t = t.cuda()
    return Variable(t, volatile=volatile)


class PackedSequencePlus(collections.namedtuple('PackedSequencePlus',
        ['ps', 'lengths', 'sort_to_orig'])):

    def apply(self, fn):
        return PackedSequencePlus(
            torch.nn.utils.rnn.PackedSequence(
                fn(self.ps.data), self.ps.batch_sizes), self.lengths,
            self.sort_to_orig)

    def with_new_ps(self, ps):
        return PackedSequencePlus(ps, self.lengths, self.sort_to_orig)

    def pad(self, batch_first, others_to_unsort=()):
        padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.ps, batch_first=batch_first)
        results = padded[
            self.sort_to_orig], [seq_lengths[i] for i in self.sort_to_orig]
        return results + tuple(t[self.sort_to_orig] for t in others_to_unsort)


def sort_lists_by_length(lists):
    # lists_sorted: lists sorted by length of each element, descending
    # orig_to_sort: tuple of integers, satisfies the following:
    #   tuple(lists[i] for i in orig_to_sort) == lists_sorted
    orig_to_sort, lists_sorted = zip(*sorted(
        enumerate(lists), key=lambda x: len(x[1]), reverse=True))
    # sort_to_orig: list of integers, satisfies the following:
    #   [lists_sorted[i] for i in sort_to_orig] == lists
    sort_to_orig = [
        x[0] for x in sorted(
            enumerate(orig_to_sort), key=operator.itemgetter(1))
    ]

    return lists_sorted, sort_to_orig


def batch_bounds_for_packing(lengths):
    '''Returns how many items in batch have length >= i at step i.

    Examples:
      [5] -> [1, 1, 1, 1, 1]
      [5, 5] -> [2, 2, 2, 2, 2]
      [5, 3] -> [2, 2, 2, 1, 1]
      [5, 4, 1, 1] -> [4, 2, 2, 2, 1]
    '''

    last_length = 0
    count = len(lengths)
    result = []
    for length, group in itertools.groupby(reversed(lengths)):
        if length <= last_length:
            raise ValueError('lengths must be decreasing and positive')
        result.extend([count] * (length - last_length))
        count -= sum(1 for _ in group)
        last_length = length
    return result


def lists_to_packed_sequence(lists, stoi, cuda, volatile):
    # Note, we are not using pack_sequence here, because it takes a list of Variables which we want to avoid since it
    # may cause memory fragmentation.

    # lists_sorted: lists sorted by length of each element, descending
    # orig_to_sort: tuple of integers, satisfies the following:
    #   tuple(lists[i] for i in orig_to_sort) == lists_sorted
    orig_to_sort, lists_sorted = zip(
            *sorted(enumerate(lists), key=lambda x: len(x[1]), reverse=True))
    # sort_to_orig: list of integers, satisfies the following:
    #   [lists_sorted[i] for i in sort_to_orig] == lists
    sort_to_orig = [x[0] for x in sorted(
            enumerate(orig_to_sort), key=operator.itemgetter(1))]

    v = numpy_to_tensor(lists_to_numpy(lists_sorted, stoi, 0), cuda, volatile)
    lens = lengths(lists_sorted)
    return PackedSequencePlus(
        torch.nn.utils.rnn.pack_padded_sequence(
            v, lens, batch_first=True),
        lens,
        sort_to_orig)


def encode_io(stoi, batch, cuda, volatile=False):
    input_keys, arg_nums, inputs, outputs = [], [], [], []
    for batch_id, example in enumerate(batch):
        for test in example.input_tests:
            arg_nums.append(len(test["input"].items()))
            for key, value in test["input"].items():
                input_keys.append(stoi(key))
                inputs.append(str(value))
            outputs.append(str(test["output"]))
    inputs = lists_to_numpy(inputs, lambda x: ord(x), 0)
    outputs = lists_to_numpy(outputs, lambda x: ord(x), 0)
    return (
        numpy_to_tensor(input_keys, cuda, volatile),
        numpy_to_tensor(inputs, cuda, volatile),
        arg_nums,
        numpy_to_tensor(outputs, cuda, volatile)
    )


def encode_schema_text(stoi, batch, cuda, volatile=False, packed_sequence=True, include_schema=True):
    texts = []
    for example in batch:
        schema = []
        if include_schema:
            for name, type_ in example.schema.args.items():
                schema.extend([name, type_, '|||'])
        texts.append(schema + example.text)
    if packed_sequence:
        return lists_to_packed_sequence(texts, stoi, cuda, volatile)
    return numpy_to_tensor(lists_to_numpy(texts, stoi, 0), cuda, volatile)


def encode_text(stoi, batch, cuda, volatile=False, packed_sequence=True):
    return encode_schema_text(stoi, batch, cuda, volatile, packed_sequence, include_schema=False)


def encode_output_text(stoi, batch, cuda, volatile=False):
    texts = [example.text for example in batch]
    return lists_padding_to_tensor(texts, stoi, cuda, volatile)


def prepare_code_sequence(batch):
    codes = []
    for example in batch:
        other_funcs = []
        if hasattr(example, 'funcs'):
            for code_func in example.funcs:
                other_funcs.append(code_func.name)
                other_funcs.extend(code_func.code_sequence)
        codes.append(other_funcs + example.code_sequence)
    return codes


def encode_input_code_seq(stoi, batch, cuda, volatile=False, packed_sequence=True):
    """Returns Tensor with code sequences from the batch for encoding."""
    codes = prepare_code_sequence(batch)
    if packed_sequence:
        return lists_to_packed_sequence(codes, stoi, cuda, volatile)
    return lists_to_packed_sequence(codes, stoi, cuda, volatile)


def lists_padding_to_tensor(lsts, stoi, cuda, volatile=False):
    max_length = max([len(lst) for lst in lsts]) + 2
    # -1: special padding value so that we don't compute the loss over it
    result = np.full((len(lsts), max_length), -1, dtype=np.int64)
    for i, lst in enumerate(lsts):
        result[i][0] = 0
        for j, word in enumerate(lst):
            result[i, j + 1] = stoi(word) if stoi is not None else word
        result[i, len(lst) + 1] = 1  # End with </S>
    return numpy_to_tensor(result, cuda, volatile)


def encode_output_code_seq(stoi, batch, cuda, volatile=False):
    """Returns Tensor with code sequences from the batch for decoding.

    The tensor is padded with -1 which are specifically used to ignore at loss computation.
    """
    codes = prepare_code_sequence(batch)
    return lists_padding_to_tensor(codes, stoi, cuda, volatile)


def encode_candidate_code_seq(stoi, batch, cuda, volatile=False):
    batch_ids = []
    code_seqs = []
    for batch_id, example in enumerate(batch):
        if example.candidate_code_sequence is None:
            continue
        batch_ids.append(batch_id)
        code_seqs.append(example.candidate_code_sequence)

    if not batch_ids:
        return batch_ids, (None, None)

    return batch_ids, lists_to_packed_sequence(code_seqs, stoi, cuda, volatile)
