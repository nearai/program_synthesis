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


def lists_to_packed_sequence(lists, stoi, cuda, volatile):
    # TODO: Use torch.nn.utils.rnn.pack_sequence in 0.4.0

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
    return torch.nn.utils.rnn.pack_padded_sequence(
        v, lengths(lists_sorted), batch_first=True), sort_to_orig


def apply_to_packed_sequence(fn, ps):
    return torch.nn.utils.rnn.PackedSequence(fn(ps.data), ps.batch_sizes)


def encode_io(vocab, batch, cuda, volatile=False):
    input_keys, input_ids, inputs, outputs = [], [], [], []
    for batch_id, example in enumerate(batch):
        for test in example.input_tests:
            for key, value in test["input"].items():
                input_keys.append(vocab.stoi(key))
                input_ids.append(batch_id)
                inputs.append(str(value))
            outputs.append(str(test["output"]))
    inputs = lists_to_numpy(inputs, lambda x: ord(x), 0)
    outputs = lists_to_numpy(outputs, lambda x: ord(x), 0)
    return (
        numpy_to_tensor(input_keys, cuda, volatile),
        numpy_to_tensor(input_ids, cuda, volatile),
        numpy_to_tensor(inputs, cuda, volatile),
        numpy_to_tensor(outputs, cuda, volatile))


def encode_schema_text(vocab, batch, cuda, volatile=False, packed_sequence=True):
    texts = []
    for example in batch:
        schema = []
        for name, type_ in example.schema.args.iteritems():
            schema.extend([name, type_, '|||'])
        texts.append(schema + example.text)
    if packed_sequence:
        return lists_to_packed_sequence(texts, vocab.stoi, cuda, volatile)
    return numpy_to_tensor(lists_to_numpy(texts, vocab.stoi, 0), cuda, volatile)


def prepare_code_sequence(batch):
    codes = []
    for example in batch:
        other_funcs = []
        for code_func in example.funcs:
            other_funcs.append(code_func.name)
            other_funcs.extend(code_func.code_sequence)
        codes.append(other_funcs + example.code_sequence)
    return codes


def encode_input_code_seq(vocab, batch, cuda, volatile=False):
    """Returns Tensor with code sequences from the batch for encoding."""
    codes = prepare_code_sequence(batch)
    return lists_to_packed_sequence(codes, vocab.stoi, cuda, volatile)


def lists_padding_to_tensor(lsts, stoi, cuda, volatile=False):
    max_length = max([len(lst) for lst in lsts]) + 2
    # -1: special padding value so that we don't compute the loss over it
    result = np.full((len(lsts), max_length), -1, dtype=np.int64)
    for i, lst in enumerate(lsts):
        result[i][0] = 0
        for j, word in enumerate(lst):
            result[i, j + 1] = stoi(word)
        result[i, len(lst) + 1] = 1  # End with </S>
    return numpy_to_tensor(result, cuda, volatile)


def encode_output_code_seq(vocab, batch, cuda, volatile=False):
    """Returns Tensor with code sequences from the batch for decoding.

    The tensor is padded with -1 which are specifically used to ignore at loss computation.
    """
    codes = prepare_code_sequence(batch)
    return lists_padding_to_tensor(codes, vocab.stoi, cuda, volatile)


def encode_candidate_code_seq(vocab, batch, cuda, volatile=False):
    batch_ids = []
    code_seqs = []
    for batch_id, example in enumerate(batch):
        if example.candidate_code_sequence is None:
            continue
        batch_ids.append(batch_id)
        code_seqs.append(example.candidate_code_sequence)

    if not batch_ids:
        return batch_ids, (None, None)

    return batch_ids, lists_to_packed_sequence(code_seqs, vocab.stoi, cuda, volatile)
