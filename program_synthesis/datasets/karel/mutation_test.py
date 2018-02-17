import unittest

import mutation

def compute_edit_ops(source_str, target_str):
    return list(mutation.compute_edit_ops(source_str, target_str, chr))


class MutationTest(unittest.TestCase):
    def test_edit_simple(self):
        self.assertEqual(
                compute_edit_ops('Levenshtein', 'Lenvinsten'),
                [(0, 'keep', None),
                    (1, 'keep', None),
                    (2, 'insert', 'n'),
                    (2, 'keep', None),
                    (3, 'replace', 'i'),
                    (4, 'keep', None),
                    (5, 'keep', None),
                    (6, 'delete', None),
                    (7, 'keep', None),
                    (8, 'keep', None),
                    (9, 'delete', None),
                    (10, 'keep', None)])



