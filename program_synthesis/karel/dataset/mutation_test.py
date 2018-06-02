import copy
import unittest

import numpy as np

from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.dataset import parser_for_synthesis


class MutationTest(unittest.TestCase):
    def setUp(self):
        self.tree = parser_for_synthesis.KarelForSynthesisParser(
            build_tree=True).parse((
                'DEF run m( REPEAT R=5 r( turnLeft IFELSE c( '
                'markersPresent c) i( turnRight i) ELSE e( move e) r) '
                'pickMarker m)').split())

    def testMutateOnce(self):
        rng = np.random.RandomState(9876)
        for i in range(100):
            mutation.mutate_n(self.tree, 1, rng=rng)

if __name__ == '__main__':
    unittest.main()
