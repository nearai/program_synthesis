import unittest
import math

from program_synthesis.algolisp.dataset.code_lisp import *
from program_synthesis.algolisp.dataset.code_lisp_parser import parse_and_compile
from program_synthesis.algolisp.dataset import code_trace


class TestLisp(unittest.TestCase):
    def test_tree_compile_basic(self):
        lisp_units = load_lisp_units()

        func = compile_statement(lisp_units, "+")
        self.assertEqual(func(1, 2), 3)

        func = compile_statement(lisp_units, ["+", "1", ["*", "2", "2"]])
        self.assertEqual(func(), 5)

        func = compile_statement(lisp_units, ["square", ["square", "2"]])
        self.assertEqual(func(), 16)

        func = compile_statement(lisp_units, ["if", "false", "+", "-"])
        self.assertEqual(func()(3, 2), 1)

    def test_compile_func(self):
        lisp_units = load_lisp_units()

        func = compile_func(lisp_units, "test", "a", [("a", Number.T)], Number.T)
        self.assertEqual(func(4), 4)

        func = compile_func(lisp_units, "test", ["+", "1", "a"], [("a", Number.T)], Number.T)
        self.assertEqual(func(4), 5)

    def test_compile_func_arrays(self):
        lisp_units = load_lisp_units()

        a = [1,2,3,4,5]
        program = ["+", ["reduce", "a", "0", "+"], "1"]
        f = compile_func(lisp_units, "test", program, [("a", Array(Number.T))], Number.T)
        self.assertEqual(f(a), 16)

        program = ["reduce", ["map", "a", "square"], "0", "+"]
        f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T))], Number.T)
        self.assertEqual(f(a), sum([x*x for x in a]))

        # program = ['map', ['filter', ['range', '0', ['len', 'a']], 'is_odd'], ['partial0', 'a', 'int-deref']]
        # test_lisp_validity(lisp_units, program, {"a": Array(Number.T)}, Array(Number.T))
        # f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T))], Array(Number.T))
        # self.assertEqual(f(a), [2, 4])

        # program = ['filter', ['filter', 'a', ['partial0', 'b', 'contains']], 'is_odd']
        # f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T)), ("b", Array(Number.T))], Array(Number.T))
        # self.assertEqual(f(a, [2, 4, 5]), [5])

        # program = ['head', ['filter', 'a', ['partial0', 'b', 'contains']]]
        # test_lisp_validity(lisp_units, program, {"a": Array(Number.T), 'b': Array(Number.T)}, Number.T)
        # f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T)), ("b", Array(Number.T))], Number.T)
        # self.assertEqual(f([2, 4, 5], [2, 10]), 2)

        program = ['head', ['slice', ['slice', 'a', 'b', 'c'], 'd', 'e']]
        test_lisp_validity(lisp_units, program, {"a": Array(Number.T), 'b': Number.T, 'c': Number.T, 'd': Number.T, 'e': Number.T}, Number.T)
        f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T)), ("b", Number.T), ("c", Number.T), ("d", Number.T), ("e", Number.T)], Number.T)
        self.assertEqual(f([2, 4, 5], 0, -1, 0, 10), 2)

        program = ['head',
            ['slice',
                ['slice', 'a', 'b', 'c'],
                ['/',
                    ['deref', ['sort', 'a'], ['/', ['len', 'a'], '2']],
                    ['len', ['slice', 'a', 'b', 'c']]],
                ['len', ['slice', 'a', 'b', 'c']]]]
        test_lisp_validity(lisp_units, program, {"a": Array(Number.T), 'b': Number.T, 'c': Number.T}, Number.T)
        f = compile_func(lisp_units, "test2", program, [("a", Array(Number.T)), ("b", Number.T), ("c", Number.T)], Number.T)
        self.assertEqual(f([2, 4, 5, 6, 7, 8], 0, 6), 4)
        with self.assertRaises(ValueError):
            self.assertEqual(f([2, 4, 5, 6, 7, 8], 3, 3), 4)

    def test_standard_funcs(self):
        lisp_units = load_lisp_units()

        self.assertTrue(compile_statement(lisp_units, "is_prime")(7))
        self.assertFalse(compile_statement(lisp_units, "is_prime")(8))

        self.assertTrue(compile_statement(lisp_units, "is_sorted")([]))
        self.assertFalse(compile_statement(lisp_units, "is_sorted")([3, 2, 3]))
        self.assertTrue(compile_statement(lisp_units, "is_sorted")([1, 2, 3]))

    def test_func_arguments(self):
        lisp_units = load_lisp_units()

        program = ["x", "1", "2"]
        f = compile_func(lisp_units, "test", program, [("x", FuncType(Number.T, [Number.T, Number.T]))], Number.T)
        self.assertEqual(f(lambda x, y: x + y), 3)

        program = "a"
        f = compile_func(lisp_units, "test", program, [("a", Number.T)], Number.T)
        self.assertEqual(f(2), 2)

    def test_combine(self):
        lisp_units = load_lisp_units()

        program = ["combine", ["partial0", "1", "+"], "*"]
        f = compile_statement(lisp_units, program)
        self.assertEqual(f()(2, 3), 7)

        program = ['reduce', ['filter', 'a', ['combine', ['combine', '!', 'is_prime'], 'square']], '0', 'max']
        test_lisp_validity(lisp_units, program, {'a': Array(Number.T)}, Number.T)
        f = compile_func(lisp_units, "test", program, [("a", Array(Number.T))], Number.T)
        self.assertEqual(f([1,2]), 2)

    def test_recursive_lambda(self):
        lisp_units = load_lisp_units()

        def fib(x):
            if x < 2:
                return 1
            return fib(x - 2) + fib(x - 1)

        program = ["invoke1", 
            ["lambda1", 
                ["if", ["<", "arg1", "2"], "1", 
                    ["+", ["self", ["-", "arg1", "2"]], 
                    ["self", ["-", "arg1", "1"]]]]],
            "x"]
        test_lisp_validity(lisp_units, program, {"x": Number.T}, Number.T, name='fib')
        f = compile_func(lisp_units, "fib", program, [("x", Number.T)], Number.T)
        self.assertEqual(f(6), fib(6))

    def test_dict(self):
        lisp_units = load_lisp_units()
        program = ["dict-new", "int", "int"]
        self.assertTrue(test_lisp_validity(
            lisp_units, program, [], Dict(Number.T, Number.T)))
        f = compile_statement(lisp_units, program)
        self.assertDictEqual(f(), {})

        # Search can't find this...
        program = ["dict-keys", ["reduce", "x", ["dict-new", "int", "int"], ["partial2", "1", "dict-insert"]]]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T)}, Array(Number.T))
        f = compile_func(lisp_units, "unique", program, [("x", Array(Number.T))], Array(Number.T))
        self.assertEqual(f([1,2,2,3,3,1,2,3]), [1,2,3])

    def test_lambda(self):
        lisp_units = load_lisp_units()
        program = ["map", "x", ["lambda1", ["+", "arg1", "arg1"]]]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T)}, Array(Number.T))
        f = compile_func(lisp_units, "map2", program, [("x", Array(Number.T))], Array(Number.T))
        self.assertEqual(f([3, 4]), [6, 8])

        program = ["map", "x", ["lambda1", ["+", "y", "arg1"]]]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T), 'y': Number.T}, Array(Number.T))
        f = compile_func(lisp_units, "map2", program, [("x", Array(Number.T)), ('y', Number.T)], Array(Number.T))
        self.assertEqual(f([3, 4], 3), [6, 7])

        program = ["reduce", "x", "0", ["lambda2", ["+", "arg1", "arg2"]]]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T), 'y': Number.T}, Number.T)
        f = compile_func(lisp_units, "reduce2", program, [("x", Array(Number.T))], Number.T)
        self.assertEqual(f([3, 4]), 7)

        program = ["+", ["head", ["map", "x", ["lambda1", ["reduce", "y", "arg1", ["lambda2", ["+", "arg1", "arg2"]]]]]], "1"]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T), 'y': Array(Number.T)}, Number.T)
        f = compile_func(lisp_units, "map2", program, [("x", Array(Number.T)), ('y', Number.T)], Number.T)
        self.assertEqual(f([3, 4], [1, 2]), 7)

        program = ["lambda1", ["if", ["<", ["deref", "a", "arg1"], ["deref", ["filter", ["reverse", "c"], ["partial0", "b", "<"]], "arg1"]], "1", "0"]]
        self.assertEqual(test_lisp_validity(lisp_units, program, {"a": Array(Number.T), 'b': Number.T, 'c': Array(Number.T)}, FuncType(Type.T, [Number.T])), FuncType(Number.T, [Number.T]))

        program = ["reduce", ["map", ["range", "0", ["min", ["len", "a"], ["len", ["filter", ["reverse", "c"], ["partial0", "b", "<"]]]]], ["lambda1", ["if", ["<", ["deref", "a", "arg1"], ["deref", ["filter", ["reverse", "c"], ["partial0", "b", "<"]], "arg1"]], "1", "0"]]], "0", "+"]
        self.assertEqual(test_lisp_validity(lisp_units, program, {"a": Array(Number.T), 'b': Number.T, 'c': Array(Number.T)}, Type.T), Number.T)
        f = compile_func(lisp_units, "deepcoder_task3", program, [("a", Array(Number.T)), ('b', Number.T), ('c', Number.T)], Number.T)
        self.assertEqual(f([26, 28, 29, 2, 19, 22, 26, 4, 11, 28, 22, 19, 8, 27], 30, [7, 28, 26, 26, 28, 16, 7, 7, 23, 15]), 0)

    def test_parse(self):
        lisp_units = load_lisp_units()
        command = """def test(int[] a):int reduce(a, 0, +)"""
        funcs = parse_and_compile(lisp_units, command)
        self.assertEqual(funcs['test']([1, 2, 3]), 6)

    def test_type_mismatch(self):
        lisp_units = load_lisp_units()
        program = ['filter', ['map', 'a', ['partial1', '*', '*']], 'is_odd']
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {"a": Array(Number.T)}, Array(Number.T))
        program = ['reduce', 'a', '0', ['head', 'a']]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {"a": Array(Number.T)}, Array(Number.T))
        program = ['map', 'a', ['partial0', '*', '>=']]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {"a": Array(Number.T)}, Array(Number.T))
        program = ['head', ['map', 'a', ['partial0', '1', '>=']]]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {"a": Array(Number.T)}, Array(Number.T))
        program = ['reduce', ['map', 'a', ['partial1', ['tail', 'b'], '*']], '0', 'max']
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T), 'b': Number.T}, Number.T)
        program = ['len', ['filter', u'a', ['partial0', 'a', ['combine', '!', 'is_prime']]]]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T), 'b': Number.T}, Number.T)
        program = ['reduce', ['map', u'a', ['deref', ['sort', 'a'], 'b']], '0', 'max']
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T), 'b': Number.T}, Number.T)
        program = ['dict-keys', ['head', ['slice', 'a', '40', '40']]]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T), 'b': Number.T}, Array(Number.T))
        # program = ['dict-keys', ['reduce', ['filter', u'a', 'is_even'], ['dict-new', 'int', 'int'], ['deref', u'a', '2']]]
        # with self.assertRaises(TypeError):
        #     test_lisp_validity(lisp_units, program, {'a': Array(Number.T)}, Array(Number.T))
        # program = ['filter', ['deref', ['array-new', 'int'], '2'], 'is_even']
        # with self.assertRaises(TypeError):
        #     test_lisp_validity(lisp_units, program, {'a': Array(Number.T)}, Array(Number.T))
        program = ['reduce', ['map', u'a', ['partial2', '+', ['head', ['map', 'a', 'head']]]], '0', 'max']
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T)}, Number.T)
        program = ['reduce', ['map', u'a', ['partial0', ['+', ['head', u'a'], u'b'], ['head', ['map', u'a', 'head']]]], '0', 'max']
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T), 'b': Number.T}, Number.T)
        program = ['reduce', 'b', '1000000000', ['partial1', '2', '*']]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'b': Array(Number.T)}, Number.T)
        program = ['deref', ['deref', ['dict-keys', ['dict-new', 'int', 'int']], 1], 2]
        with self.assertRaises(TypeError):
            test_lisp_validity(lisp_units, program, {'a': Array(Number.T)}, Number.T)

    def test_runtime_issues(self):
        lisp_units = load_lisp_units()
        f = compile_statement(lisp_units, ["/", "1", "0"])
        with self.assertRaises(ValueError):
            f()

    def test_traces(self):
        lisp_units = load_lisp_units()
        t = code_trace.CodeTrace()
        f = compile_statement(lisp_units, ["/", 4, "2"], trace=t)
        f()
        self.assertEqual(
            t.history, [('/', [4, 2])]
        )
        self.assertEqual(
            t.results, [2]
        )
        t.clear()
        program = ["reduce", "x", "0", ["lambda2", ["+", "arg1", "arg2"]]]
        test_lisp_validity(lisp_units, program, {"x": Array(Number.T), 'y': Number.T}, Number.T)
        f = compile_func(lisp_units, "reduce2", program, [("x", Array(Number.T))], Number.T, trace=t)
        self.assertEqual(f([3, 4]), 7)
        self.assertEqual(
            t.history, [('reduce', [[3, 4], 0, 'LAMBDA1']), ('+', [0, 3]), ('+', [3, 4])]
        )
        self.assertEqual(
            t.results, [7, 3, 7]
        )

    def test_choice(self):
        lisp_units = load_specification_units()
        funcs = parse_and_compile(lisp_units,
                                  "def test(int[] x):int len(choice((i, j, k), crossproduct(crossproduct(range(0, len(x)), range(0, len(x))), range(0, len(x))), &&(&&(&&(<(i, j), <(j, k)), !=(k,i)), >(+(deref(x, i), deref(x, j)), deref(x, k))), 1))")
        self.assertEqual(funcs['test']([2, 2, 3]), 1)
        funcs = parse_and_compile(lisp_units,
                                  "def test(string[] s, string s1):int if(contains(s, s1), str_concat(s1, str(reduce(choice(x, range(0, 40), !(contains(s, str_concat(s1, str(x)))), x), inf, min))), s1)")
        self.assertEqual(funcs['test'](["xyz"], "abc"), "abc")
        self.assertEqual(funcs['test'](["abc", "abc0", "abc1", "abc2"], "abc"), "abc3")


if __name__ == "__main__":
    unittest.main()
