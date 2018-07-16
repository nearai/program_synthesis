import collections

import numpy as np

import torch


class PackedSequencePlus(collections.namedtuple('PackedSequencePlus',
                                                ['ps', 'lengths', 'sort_to_orig', 'orig_to_sort'])):

    def __new__(cls, ps, lengths, sort_to_orig, orig_to_sort):
        sort_to_orig = np.array(sort_to_orig)
        orig_to_sort = np.array(orig_to_sort)
        self = super(PackedSequencePlus, cls).__new__(
            cls, ps, lengths, sort_to_orig, orig_to_sort)
        self.cum_batch_sizes = np.cumsum([0] + self.ps.batch_sizes.tolist()[:-1])
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
        results = padded[self.sort_to_orig, :], [seq_lengths[i]
                                                 for i in self.sort_to_orig]
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
