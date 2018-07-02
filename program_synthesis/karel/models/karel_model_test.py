import unittest
import sys

import numpy as np

from program_synthesis.karel.models import karel_model
from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import dataset
from program_synthesis.karel.dataset import executor


def make_karel_example(code_sequence, input_value, ref_example=None):
    return dataset.KarelExample(
        idx=0,
        guid='',
        code_sequence=tuple(code_sequence),
        input_tests=[
            {'input': [i, input_value], 'output': [0]} for i in range(5)
        ],
        tests=[{'input': [5, input_value], 'output': [0]}],
        ref_example=ref_example)


def indices(array):
    return np.where(array.ravel())[0].tolist()


class Args(object):
    def __init__(self, **args):
        for k, v in args.items():
            setattr(self, k, v)


class RefineBatchProcessorTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            make_karel_example(
                ['w', 'a', 'b', 'x'], 100,
                ref_example=make_karel_example(['a', 'b'], 1000)),
            make_karel_example(
                ['y'], 200, ref_example=make_karel_example(['c'], 1001)),
            make_karel_example(
                ['d', 'z', 'f'], 300,
                ref_example=make_karel_example(['d', 'e', 'f'], 1002)),
        ]
        self.vocab = data.PlaceholderVocab({
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'w': 6,
            'x': 7,
            'y': 8,
            'z': 9,
            '</S>': 10
        }, num_placeholders=0)

    def test_compute_edit_ops(self):
        args = Args(
            karel_code_enc='default',
            karel_refine_dec='edit',
            karel_trace_enc='none')

        bp = karel_model.KarelLGRLRefineBatchProcessor(args, self.vocab, False)
        out = bp(self.examples)
        dec_data = out.dec_data

        # Check that I/O grids are as expected
        io_embed_flat = out.input_grids.data.view(15, 15, 18, 18).numpy()
        for i in range(3):
            for j in range(i * 5, i * 5 + 5):
                self.assertEqual(
                    indices(io_embed_flat[j]), [j % 5, (i + 1) * 100])

        # abA / cB / defC -> defC / abA / cB -> dacebBfAC
        #                                       012345678
        # wabxA / yB / dzfC -> wabxA / dzfC / yB -> wdyazBbfxCA
        #
        # ops:
        # <s>, insert w, keep, keep, insert x
        # 0, 4+6*2=16, 2, 2, 4+7*2=18, eos
        # <s>, keep, replace z, keep
        # 0, 2, 5+9*2=23, 2, eos
        # <s>, replace y
        # 0, 5+8*2=21, eos
        self.assertEqual(dec_data.input.ps.data.data.numpy()[:, 0].tolist(),
                [0, 0, 0, 16, 2, 21, 2, 23, 2, 2, 18])
        self.assertEqual(dec_data.output.ps.data.data.numpy()[:, 0].tolist(),
                [16, 2, 21, 2, 23, 1, 2, 2, 18, 1, 1])

        # emb_pos:
        # w -> a, a -> a, b -> b, x -> A, A -> A
        # y -> c, B -> B
        # d -> d, z -> e, f -> f, C -> C
        self.assertEqual(dec_data.input.ps.data.data.numpy()[:, 1].tolist(),
                [1, 0, 2, 1, 3, 5, 4, 6, 7, 8, 7])

        # last_token
        # 000wdyazbfx -> stoi
        # XXX first 3 elements should really be 0
        self.assertEqual(dec_data.input.ps.data.data.numpy()[:, 2].tolist(),
              [2, 2, 2, 6, 3, 8, 0, 9, 1, 5, 7])

        # io_embed_indices
        io_embed_indices = dec_data.io_embed_indices.numpy()
        self.assertEqual(len(io_embed_indices), 5 * len('wdyazBbfxCA'))

        # Check that we are getting the right input from io_embed_flat.
        # 0, 2, 1, ... are the original indices of the sorted sequences.
        i = 0
        for orig_batch_i in [0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 0]:
            for j in range(orig_batch_i * 5, orig_batch_i * 5 + 5):
                self.assertEqual(
                    indices(io_embed_flat[io_embed_indices[i]]),
                    [j % 5, (orig_batch_i + 1) * 100])
                i += 1
        self.assertEqual(i, len(io_embed_indices))


if __name__ == '__main__':
    unittest.main()
