import unittest

from program_synthesis.karel.models import prepare_spec


class PackedSequenceTest(unittest.TestCase):

    @unittest.skip('TransposedPackedSequence is missing')
    def test_transpose1(self):
        psp = prepare_spec.lists_to_packed_sequence(
            [[0, 1], [2], [3, 4, 5], [6, 7, 8]], lambda x: x, False, False)
        psp_t = psp.transpose()
        self.assertEqual(
            psp_t.data.data.numpy().tolist(), [3, 4, 5, 6, 7, 8, 0, 1, 2])
        self.assertEqual(psp_t.lengths, [3, 3, 2, 1]),
        self.assertEqual(psp_t.sort_to_orig, [2, 3, 0, 1])
        self.assertEqual(psp_t.orig_to_sort, (2, 3, 0, 1))

    @unittest.skip('TransposedPackedSequence is missing')
    def test_transpose2(self):
        psp = prepare_spec.lists_to_packed_sequence(
            [[0, 1], [2], [3, 4, 5]], lambda x: x, False, False)
        psp_t = psp.transpose()
        self.assertEqual(
            psp_t.data.data.numpy().tolist(), [3, 4, 5, 0, 1, 2])
        self.assertEqual(psp_t.lengths, [3, 2, 1]),
        self.assertEqual(psp_t.sort_to_orig, [1, 2, 0])
        self.assertEqual(psp_t.orig_to_sort, (2, 0, 1))

        psp_t = psp_t.expand(3)
        self.assertEqual(psp_t.data.data.numpy().tolist(), [
            3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 0, 1, 0, 1, 2, 2, 2
        ])
        self.assertEqual(psp_t.lengths, [3, 3, 3,  2, 2, 2,  1, 1, 1]),
        self.assertEqual(psp_t.sort_to_orig, [3, 4, 5, 6, 7, 8, 0, 1, 2])
        self.assertEqual(psp_t.orig_to_sort, [6, 7, 8, 0, 1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()
