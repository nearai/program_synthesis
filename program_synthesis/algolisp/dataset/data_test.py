import random
import unittest

import data


class DataTest(unittest.TestCase):

    def test_flatten_code(self):
        self.assertEqual(
            data.flatten_code(["reduce", ["square", "a"], "+", "1000"], 'lisp'), 
            ["(", "reduce", "(", "square", "a", ")", "+", "1000", ")"])

    def test_unflatten_code(self):
        self.assertEqual(
            data.unflatten_code(["(", "reduce", "(", "square", "a", ")", "+", "1000", ")"], 'lisp'),
            (["reduce", ["square", "a"], "+", "1000"], True)
        )

    def test_tokenizer(self):
        self.assertEqual(
            data.tokenize_text_line("how to compute `a`-`b`<`c` where 5<c<10"),
            ["how", "to", "compute", "a", "-", "b", "<",
                "c", "where", "5", "<", "c", "<", "10"]
        )

    def test_code_tokenizer(self):
        self.assertEqual(data.tokenize_code_line("blah123qq_qq"), ["blah123qq_qq"])
        self.assertEqual(
            data.tokenize_code_line("""#include<std.io> void Main() { int blah123qq_qq[] = {}; cout << "q'q``"; _qq = False; qwe1._123 = 123; __global__ = -1.2; }"""),
            ["#", "include", "<", "std", ".", "io", ">", "void", "Main", "(", ")", "{", "int", "blah123qq_qq", "[", "]",
             "=", "{", "}", ";", "cout", "<<", "\"", "q", "'", "q", "`", "`", "\"", ";", "_qq", "=", "False", ";", "qwe1", ".", "_123", 
             "=", "123", ";", "__global__", "=", "-1.2", ";", "}"]
        )

    def test_word_code_vocab(self):
        random.seed(42)
        word_vocab = {'<S>': 0, '</S>': 1, '<UNK>': 2, 'test': 3, 'qq': 4, 'ww': 5}
        code_vocab = {'<S>': 0, '</S>': 1, '<UNK>': 2, 'var1': 3, 'var2': 4}
        v = data.WordCodeVocab(word_vocab, code_vocab, 3)
        v.reset()
        self.assertEqual(v.wordtoi('qq'), 4)
        self.assertEqual(v.wordtoi('ww'), 5)
        self.assertEqual(v.codetoi('var1'), 3)
        self.assertEqual(v.codetoi('var2'), 4)
 
        self.assertEqual(v.wordtoi('25'), 8)
        self.assertEqual(v.codetoi('25'), 7)
        self.assertEqual(v.wordtoi('1'), 7)
        self.assertEqual(v.codetoi('1'), 6)
        self.assertEqual(v.wordtoi('3'), 6)
        self.assertEqual(v.codetoi('3'), 5)
        self.assertEqual(v.codetoi('7'), 2)
        self.assertEqual(v.itoword(8), '25')
        self.assertEqual(v.itocode(7), '25')
        self.assertEqual(v.itocode(2), '<UNK>')


if __name__ == "__main__":
    unittest.main()
