import unittest

from parameterized import parameterized
import torch
from torch.autograd import Variable
import torch.nn as nn

import slicing_joining


class TestRunRNN(unittest.TestCase):

    def test_indices(self):
        sequence = [40, 20, 50, 30, 10]
        f, r = slicing_joining._forward_and_reverse_indices(sequence)
        self.assertListEqual([2, 0, 3, 1, 4], f)
        self.assertListEqual([1, 3, 0, 2, 4], r)

    @parameterized.expand([
        ("one_directional", False, 1),
        ("bidirectional", True, 1),
        ("one_directional_2layers", False, 2),
        ("bidirectional_2layers", True, 2),
    ])
    def test_run_rnn(self, unused_name, bidirectional, num_layers):
        def _array_to_tensor(arr):
            t = torch.FloatTensor(arr)
            return Variable(t)

        non_padded_seqs = [
            [1],
            [4, 2, 3, 8],
            [2, 0, 3],
            [3, 4, 2, 1, 3],
            [2, 3, 4],
        ]
        non_padded_seqs = [_array_to_tensor(s).unsqueeze(1) for s in non_padded_seqs]
        padded_seqs = [
            [1, -100, -100, -100, -100],
            [4, 2, 3, 8, -100],
            [2, 0, 3, -100, -100],
            [3, 4, 2, 1, 3],
            [2, 3, 4, -100, -100],
        ]
        padded_seqs = _array_to_tensor(padded_seqs).unsqueeze(2)
        seq_lengths = [1, 4, 3, 5, 3]
        net = nn.GRU(input_size=1, hidden_size=4, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        total_layers = (2 if bidirectional else 1) * num_layers
        init_state = Variable(torch.zeros(total_layers, 5, 4))
        slicing_joining.run_padded_rnn(padded_seqs, net, init_state, seq_lengths, cuda=False)
        slicing_joining.run_rnn(non_padded_seqs, net, init_state, cuda=False)


if __name__ == "__main__":
    unittest.main()
