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


class PackedSequencePlus(collections.namedtuple('PackedSequencePlus',
        ['ps', 'lengths', 'sort_to_orig', 'orig_to_sort'])):
    def __new__(cls, ps, lengths, sort_to_orig, orig_to_sort):
        sort_to_orig = np.array(sort_to_orig)
        orig_to_sort = np.array(orig_to_sort)
        self = super(PackedSequencePlus, cls).__new__(
            cls, ps, lengths, sort_to_orig, orig_to_sort)
        self.cum_batch_sizes = np.cumsum([0] + self.ps.batch_sizes[:-1])
        return self

    def apply(self, fn):
        return PackedSequencePlus(
            torch.nn.utils.rnn.PackedSequence(
                fn(self.ps.data), self.ps.batch_sizes), self.lengths,
            self.sort_to_orig,
            self.orig_to_sort)

    def with_new_ps(self, ps):
        return PackedSequencePlus(ps, self.lengths, self.sort_to_orig,
                self.orig_to_sort)

    def pad(self, batch_first, others_to_unsort=(), padding_value=0.0):
        padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.ps, batch_first=batch_first, padding_value=padding_value)
        results = padded[self.sort_to_orig, :], [seq_lengths[i] for i in self.sort_to_orig]
        return results + tuple(t[self.sort_to_orig] for t in others_to_unsort)

    def cuda(self, async=False):
        if self.ps.data.is_cuda:
            return self
        return self.apply(lambda d: d.cuda(async=async))

    def raw_index(self, orig_batch_idx, seq_idx):
        result = np.take(self.cum_batch_sizes, seq_idx) + np.take(
                self.sort_to_orig, orig_batch_idx)
        if self.ps.data is not None:
            assert np.all(result < len(self.ps.data))
        return result

    def select(self, orig_batch_idx, seq_idx):
        return self.ps.data[self.raw_index(orig_batch_idx, seq_idx)]

    def orig_index(self, raw_idx):
        seq_idx = np.searchsorted(
                self.cum_batch_sizes, raw_idx, side='right') - 1
        batch_idx = raw_idx - self.cum_batch_sizes[seq_idx]
        orig_batch_idx = self.sort_to_orig[batch_idx]
        return orig_batch_idx, seq_idx

    def orig_batch_indices(self):
        result = []
        for bs in self.ps.batch_sizes:
            result.extend(self.orig_to_sort[:bs])
        return np.array(result)

    def orig_lengths(self):
        for sort_idx in self.sort_to_orig:
            yield self.lengths[sort_idx]

    def expand(self, k):
        # Conceptually, this function does the following:
        #   Input: d1 x ...
        #   Output: d1 * k x ... where
        #     out[0] = out[1] = ... out[k],
        #     out[k + 0] = out[k + 1] = ... out[k + k],
        #   and so on.
        v = self.ps.data
        ps_data = v.unsqueeze(1).repeat(1, k, *(
            [1] * (v.dim() - 1))).view(-1, *v.shape[1:])
        batch_sizes = (np.array(self.ps.batch_sizes) * k).tolist()
        lengths = np.repeat(self.lengths, k).tolist()
        sort_to_orig = [
            exp_i for i in self.sort_to_orig for exp_i in range(i * k, i * k + k)
        ]
        orig_to_sort = [
            exp_i for i in self.orig_to_sort for exp_i in range(i * k, i * k + k)
        ]
        return PackedSequencePlus(
                torch.nn.utils.rnn.PackedSequence(ps_data, batch_sizes),
                lengths, sort_to_orig, orig_to_sort)

    def cpu(self):
        if not self.ps.data.is_cuda:
            return self
        return self.apply(lambda d: d.cpu())


def sort_lists_by_length(lists):
    # lists_sorted: lists sorted by length of each element, descending
    # orig_to_sort: tuple of integers, satisfies the following:
    #   tuple(lists[i] for i in orig_to_sort) == lists_sorted
    #   lists[orig_to_sort[sort_idx]] == lists_sorted[sort_idx]
    orig_to_sort, lists_sorted = zip(*sorted(
        enumerate(lists), key=lambda x: len(x[1]), reverse=True))
    # sort_to_orig: list of integers, satisfies the following:
    #   [lists_sorted[i] for i in sort_to_orig] == lists
    #   lists_sorted[sort_to_orig[orig_idx]] == lists[orig_idx]
    sort_to_orig = [
        x[0] for x in sorted(
            enumerate(orig_to_sort), key=operator.itemgetter(1))
    ]

    return lists_sorted, sort_to_orig, orig_to_sort


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
    for i, (length, group) in enumerate(itertools.groupby(reversed(lengths))):
        # TODO: Check that things don't blow up when some lengths are 0
        if i > 0 and  length <= last_length:
            raise ValueError('lengths must be decreasing and positive')
        result.extend([count] * (length - last_length))
        count -= sum(1 for _ in group)
        last_length = length
    return result


PSPInterleaveInfo = collections.namedtuple('PSPInterleaveInfo',
        ['input_indices', 'output_indices', 'psp_template'])


def prepare_interleave_packed_sequences(psps, interleave_indices):
    # psps: list of PackedSequencePluses.
    #       Each PackedSequencePlus has batch size x seq length items.
    # interleave_indices: list of list of psp_idx, indicating from which psp
    #                     the next element should originate.
    #   len(interleave_indices) = len(psps)
    #   interleave_indices[batch_idx][j] == psp_idx
    #   <--> result[batch_idx, j] = psps[psp_idx][batch_idx,
    #      np.sum(interleave_indices[batch_idx][:j] == psp_idx)]
    #
    # Precondition:- all PSPs have same type and item shape
    #
    # Output: a result computed by
    # result[output_indices[i]] = psps[i].ps.data[input_indices[i]]
    output_indices = [[] for _ in psps]
    input_indices = [[] for _ in psps]

    # combined_lengths: length of each sequence, in original batch order.
    #combined_lengths = np.sum(
    #        [[psp.lengths[i] for i in psp.sort_to_orig] for psp in psps],
    #        axis=0)
    combined_lengths = [len(idxs) for idxs in interleave_indices]

    orig_to_sort, sorted_lengths = zip(*sorted(
        enumerate(combined_lengths), key=operator.itemgetter(1), reverse=True))
    sort_to_orig = [
        x[0] for x in sorted(
            enumerate(orig_to_sort), key=operator.itemgetter(1))
    ]
    batch_bounds = batch_bounds_for_packing(sorted_lengths)
    interleave_iters = [iter(lst) for lst in interleave_indices]

    # idx: current cursor into result
    # i: distance from begining of sequence
    write_idx = 0
    read_idx = [[0] * len(psps) for _ in sorted_lengths]
    try:
        for i, bound in enumerate(batch_bounds):
            for  j, orig_batch_idx in enumerate(orig_to_sort[:bound]):
                # Figure out which PSP we should get
                psp_idx = next(interleave_iters[orig_batch_idx])
                # Record that we should write the value from this PSP here
                output_indices[psp_idx].append(write_idx)
                current_idx = read_idx[orig_batch_idx][psp_idx]
                # Figure out what current_idx corresponds to inside the PSP.
                assert current_idx < psps[psp_idx].lengths[j]
                input_indices[psp_idx].append(
                        int(psps[psp_idx].raw_index(orig_batch_idx,
                            current_idx)))
                read_idx[orig_batch_idx][psp_idx] += 1
                write_idx += 1
    except StopIteration:
        raise Exception('interleave_indices[{}] ended early'.format(
            orig_batch_idx))

    # Check that all of interleave_indices have been used
    for it in interleave_iters:
        ended = False
        try:
            next(it)
        except StopIteration:
            ended = True
        assert ended

    input_indices = [torch.LongTensor(t) for t in input_indices]
    output_indices = [torch.LongTensor(t) for t in output_indices]

    return PSPInterleaveInfo(input_indices, output_indices,
            PackedSequencePlus(
                torch.nn.utils.rnn.PackedSequence(None, batch_bounds),
                sorted_lengths,
                sort_to_orig,
                orig_to_sort))


def execute_interleave_psps(psps, interleave_info):
    result = Variable(psps[0].ps.data.data.new(
        sum(psp.ps.data.shape[0] for psp in psps), *psps[0].ps.data.shape[1:]))

    for out_idx, inp_idx, psp in zip(interleave_info.output_indices,
            interleave_info.input_indices, psps):
        if psp.ps.data.is_cuda:
            inp_idx = inp_idx.cuda()
            out_idx = out_idx.cuda()
        # psp.ps.data is a torch.autograd.Variable (despite its name)
        result[out_idx] = psp.ps.data[inp_idx]

    return interleave_info.psp_template.apply(lambda _: result)


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

    lists_sorted, sort_to_orig, orig_to_sort = sort_lists_by_length(lists)

    v = numpy_to_tensor(lists_to_numpy(lists_sorted, stoi, 0), cuda, volatile)
    lens = lengths(lists_sorted)
    return PackedSequencePlus(
        torch.nn.utils.rnn.pack_padded_sequence(
            v, lens, batch_first=True),
        lens,
        sort_to_orig,
        orig_to_sort)


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
