import unittest

import torch
import numpy as np
from program_synthesis.naps.examples.seq2seq import pointer_seq2seq
from torch.autograd import Variable


class TestPointerSeq2Seq(unittest.TestCase):

    def test_pointer_seq2seq(self):
        padded_texts = Variable(torch.LongTensor(
            [[0, 4, 5, 4, 1, -1],
             [0, 5, 5, 1, -1, -1],
             [0, 6, 2, 1, -1, -1]]
        ))
        extended_vocab_size = 7
        attentions = Variable(torch.FloatTensor(
                [[[0.1, 0.2, 0.1, 0.1, 0.1, 0.4],
                  [0.0, 0.2, 0.0, 0.2, 0.1, 0.5],
                  [0.1, 0.1, 0.2, 0.2, 0.1, 0.3],
                  [0.2, 0.1, 0.0, 0.2, 0.1, 0.4],
                  [0.1, 0.0, 0.2, 0.2, 0.0, 0.5],
                  [0.0, 0.0, 0.2, 0.1, 0.1, 0.6]],

                 [[0.2, 0.0, 0.0, 0.2, 0.2, 0.4],
                  [0.0, 0.0, 0.0, 0.1, 0.1, 0.8],
                  [0.0, 0.0, 0.2, 0.1, 0.2, 0.5],
                  [0.2, 0.2, 0.1, 0.0, 0.2, 0.3],
                  [0.1, 0.0, 0.1, 0.1, 0.0, 0.7],
                  [0.0, 0.2, 0.0, 0.2, 0.1, 0.5]],

                 [[0.0, 0.2, 0.2, 0.1, 0.0, 0.5],
                  [0.0, 0.1, 0.2, 0.0, 0.1, 0.6],
                  [0.1, 0.2, 0.2, 0.1, 0.1, 0.3],
                  [0.2, 0.0, 0.2, 0.2, 0.1, 0.3],
                  [0.2, 0.2, 0.0, 0.2, 0.0, 0.4],
                  [0.1, 0.0, 0.1, 0.2, 0.1, 0.5]]]))
        actual = pointer_seq2seq.location_logistics(attentions, padded_texts, extended_vocab_size)

        expected = np.array(
            [[[0.1, 0.5, 0.0, 0.0, 0.3, 0.1, 0.0],
              [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0],
              [0.1, 0.4, 0.0, 0.0, 0.3, 0.2, 0.0],
              [0.2, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0],
              [0.1, 0.5, 0.0, 0.0, 0.2, 0.2, 0.0],
              [0.0, 0.7, 0.0, 0.0, 0.1, 0.2, 0.0]],
             [[0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
              [0.2, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0],
              [0.1, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0],
              [0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0]],
             [[0.0, 0.6, 0.2, 0.0, 0.0, 0.0, 0.2],
              [0.0, 0.7, 0.2, 0.0, 0.0, 0.0, 0.1],
              [0.1, 0.5, 0.2, 0.0, 0.0, 0.0, 0.2],
              [0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0],
              [0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.2],
              [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0]]])
        np.testing.assert_almost_equal(actual.data.numpy(), expected, decimal=3)

if __name__ == "__main__":
    unittest.main()