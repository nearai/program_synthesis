import unittest

from program_synthesis.algolisp.dataset.code_types import *


class TypeTest(unittest.TestCase):

    def test_type_match(self):
        int_ = TypeType(Number.T)
        self.assertTrue(int_.compatible(TypeType(Number.T)))
        self.assertFalse(int_.compatible(TypeType(String.T)))
        self.assertTrue(AnyTypeType.T.compatible(int_))

        func_type = FuncType(return_type=Number.T, argument_types=[Number.T, Number.T])
        self.assertFalse(NotFuncType.T.compatible(func_type))
        self.assertFalse(func_type.compatible(NotFuncType.T))
        self.assertTrue(Number.T.compatible(NotFuncType.T))

        self.assertFalse(AnyFuncType.T.compatible(NotFuncType.T))
        self.assertFalse(NotFuncType.T.compatible(AnyFuncType.T))


if __name__ == "__main__":
    unittest.main()
