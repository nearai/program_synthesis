import unittest

import bleu


class TestBleu(unittest.TestCase):

    def test_bleu_multi_reference(self):
        hypothesis = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
                      'ensures', 'that', 'the', 'military', 'always',
                      'obeys', 'the', 'commands', 'of', 'the', 'party']
        refa = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
                'ensures', 'that', 'the', 'military', 'will', 'forever',
                'heed', 'Party', 'commands']
        refb = ['It', 'is', 'the', 'guiding', 'principle', 'which',
                'guarantees', 'the', 'military', 'forces', 'always',
                'being', 'under', 'the', 'command', 'of', 'the', 'Party']
        refc = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
                'army', 'always', 'to', 'heed', 'the', 'directions',
                'of', 'the', 'party']
        references = [refa, refb, refc]
        score = bleu.compute_bleu([references], [hypothesis])
        self.assertAlmostEqual(score, 0.50456667)


if __name__ == "__main__":
    unittest.main()
