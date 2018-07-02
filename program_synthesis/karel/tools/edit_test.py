import unittest

from program_synthesis.karel.tools import edit


class EditTest(unittest.TestCase):
    data = [
        ('', '', []),
        ('', 'ab', [(0, 'insert', 'a'), (0, 'insert', 'b')]),
        ('c', 'abc',
         [(0, 'insert', 'a'), (0, 'insert', 'b'), (0, 'keep', None)]),
        ('a', 'abc',
         [(0, 'keep', None), (1, 'insert', 'b'), (1, 'insert', 'c')]),
        ('Levenshtein', 'Lenvinsten',
         [(0, 'keep', None), (1, 'keep', None), (2, 'insert', 'n'),
          (2, 'keep', None), (3, 'replace', 'i'), (4, 'keep', None),
          (5, 'keep', None), (6, 'delete', None), (7, 'keep', None),
          (8, 'keep', None), (9, 'delete', None), (10, 'keep', None)]),
    ]

    def test_compute_edit_ops(self):
        for source, target, ops in self.data:
            self.assertEqual(
                list(edit.compute_edit_ops(source, target, ord)), ops)

    def test_apply_edit_ops(self):
        for source, target, ops in self.data:
            self.assertEqual(
                ''.join(edit.apply_edit_ops(source, ops)), target)


if __name__ == '__main__':
    unittest.main()
