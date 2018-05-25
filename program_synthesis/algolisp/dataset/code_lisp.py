import math
import functools
import copy
import six
import functools

from program_synthesis.algolisp.dataset.code_types import *


class ArgType(object):

    def __init__(self, base_type=None, compute=None):
        self.base_type = base_type
        self.compute = compute

    def __call__(self, r, args):
        return self.compute(r, args)


ARGS_CALL = 0
ARGS_COMPILE_ONLY = 1
ARGS_PASS_LISP = 2

MAX_STACK_DEPTH = 110


class Unit(object):

    def __init__(
            self, name, description, args, return_type, compute,
            computed_return_type=None, arguments_pass_type=ARGS_CALL):
        self.name = name
        self.description = description
        self.args = args
        self.return_type = return_type
        self.compute = compute
        self.computed_return_type = computed_return_type
        self.arguments_pass_type = arguments_pass_type

    def __repr__(self):
        return "%s(%s):%s" % (self.name, ",".join(
            [str(arg) if not callable(arg) else "computed()" for arg in self.args]), self.return_type)


class LispError(ValueError):
    pass


def if_(context, cond_, then_, else_):
    assert cond_[1]
    if cond_[0]():
        return then_[0]() if then_[1] else then_[0]
    else:
        return else_[0]() if else_[1] else else_[0]


def deref(a, idx):
    if idx < 0 or idx >= len(a):
        raise ValueError("Index %s out of bounds (array length is %s)" % (idx, len(a)))
    return a[idx]

def dict_insert(d, key, value):
    d[key] = value
    return d


def head(a):
    if len(a) == 0:
        raise ValueError("Computing HEAD of an empty array")
    return a[0]


def range_func(lo, hi):
    if hi - lo > 1000:
        raise LispError("Currently range only runs for less then 1000 elements.")
    return range(int(lo), int(hi))


def reduce_func(collection, init, func):
    return functools.reduce(func, collection, init)


def partial(index, value, func):
    def _wrapper(*args):
        args = args[:index] + (value,) + args[index:]
        return func(*args)
    return _wrapper


def lambda1(context, body):
    assert body[1]
    def lambda_func(arg1):
        if len(context.stack) > MAX_STACK_DEPTH:
            raise ValueError("Stack Overflow")
        context.enter_scope(closure=True)
        context['arg1'] = arg1
        context['self'] = lambda_func
        res = body[0]()
        context.exit_scope()
        return res    
    return lambda_func


def lambda2(context, body):
    assert body[1]
    def lambda_func(arg1, arg2):
        if len(context.stack) > MAX_STACK_DEPTH:
            raise ValueError("Stack Overflow")
        context.enter_scope(closure=True)
        context['arg1'] = arg1
        context['arg2'] = arg2
        context['self'] = lambda_func
        res = body[0]()
        context.exit_scope()
        return res    
    return lambda_func


def div(x, y):
    if y == 0:
        raise ValueError("Division by zero")
    if isinstance(x, int):
        return x // y
    return x / y


def slice_func(arr, lo, hi):
    lo = max(0, lo)
    return arr[lo:hi]


def load_default_lisp_units():
    units = [
        Unit("int", "Integer type", [], TypeType(Number.T), lambda: int),
        Unit("!", "Negates the boolean input", [Boolean.T], Boolean.T, lambda x: not x),
        Unit("+", "Sums two numbers", [Number.T, Number.T], Number.T, lambda x, y: x + y),
        Unit("-", "Subtracts two numbers", [Number.T, Number.T], Number.T, lambda x, y: x - y),
        Unit("*", "Multiplies two numbers", [Number.T, Number.T], Number.T, lambda x, y: x * y),
        Unit("/", "Divides two numbers", [Number.T, Number.T], Number.T, div),
        Unit("%", "Computes the remainder", [Number.T, Number.T], Number.T, lambda x, y: x % y),
        Unit("pow", "Computes number of multiplication of number given number of times", [Number.T, Number.T], Number.T, lambda x, y: x ** y),
        Unit("sqrt", "Computes a square root of a given number", [Number.T], Number.T, lambda x: math.sqrt(x)),
        Unit("floor", "Rounds down the number", [Number.T], Number.T, lambda x: int(float(x))),

        Unit("0", "0", [], Number.T, lambda: 0),
        Unit("1", "1", [], Number.T, lambda: 1),
        Unit("2", "2", [], Number.T, lambda: 2),
        Unit("10", "10", [], Number.T, lambda: 10),
        Unit("40", "40", [], Number.T, lambda: 40),
        Unit("1000000000", "1000000000", [], Number.T, lambda: 1000000000),
        Unit("inf", "infinity", [], Number.T, lambda: float('inf')),

        Unit('""', 'Empty character', [], String.T, lambda: ""),
        Unit('" "', 'Space character', [], String.T, lambda: " "),
        Unit('"z"', 'character z', [], String.T, lambda: "z"),

        Unit("false", 'Boolean value representing the condition doesn\'t hold', [], Boolean.T, lambda: False),
        Unit("true", 'Boolean value representing the condition holds', [], Boolean.T, lambda: True),
        Unit("||", "Computes logical OR of two booleans", [Boolean.T, Boolean.T], Boolean.T, lambda x, y: x or y),
        Unit("&&", "Computes logical AND of two booleans", [Boolean.T, Boolean.T], Boolean.T, lambda x, y: x and y),
        Unit("<", "Check if one element is smaller than another", [NotFuncType.T, NotFuncType.T], Boolean.T, lambda x, y: x < y),
        Unit(">", "Check if one element is greater than another", [NotFuncType.T, NotFuncType.T], Boolean.T, lambda x, y: x > y),
        Unit("<=", "Check if one element is smaller or equal than another", [NotFuncType.T, NotFuncType.T], Boolean.T, lambda x, y: x <= y),
        Unit(">=", "Check if one element is greater or equal than another", [NotFuncType.T, NotFuncType.T], Boolean.T, lambda x, y: x >= y),
        Unit("==", "Check if two elements are equal", [Type.T, Type.T], Boolean.T, lambda x, y: x == y),
        Unit("!=", "Check if two elements are not equal", [Type.T, Type.T], Boolean.T, lambda x, y: x != y),

        Unit("if", "Condition", [
            Boolean.T, (lambda r, args: r), (lambda r, args: args[1])], AnyType.T,
            if_, lambda r, args: args[1], ARGS_COMPILE_ONLY),

        Unit("reduce", "Folds the given array using the given func", 
            [Array(Type.T), 
            (lambda r, args: r), 
            (lambda r, args: FuncType(args[1], [args[1], args[0].underlying_type]))],
            AnyType.T, reduce_func,
            lambda r, args: args[2].return_type),
        Unit("map", "Returns an array where each element is computed by applying the given function to the corresponding element of the given array",
            [Array(Type.T), 
             (lambda r, args: FuncType(r.underlying_type if isinstance(r, Array) else Type.T, [args[0].underlying_type]))],
            Array(AnyType.T),
            lambda collection, func: list(map(func, collection)),
            lambda r, args: Array(args[1].return_type)),
        Unit("filter", "Returns an array that contains elements of the given array for which the given function returns True",
             [(lambda r, args: r if isinstance(r, Array) else Array(Type.T)),
              (lambda r, args: FuncType(Boolean.T, [args[0].underlying_type]))],
              Array(AnyType.T), lambda arr, func: list(filter(func, arr)),
              lambda r, args: args[0]),
        Unit("range", "Returns a range of elements from lo to hi", 
            [Number.T, Number.T], Array(Number.T), range_func),

        # Functional programming tools.
        Unit("invoke1", "Invokes given function with one argument",
            [lambda r, args: FuncType(r, [Number.T]), 
             lambda r, args: args[0].argument_types[0]], AnyType.T,
            lambda f, arg: f(arg),
            lambda r, args: args[0].return_type),
        Unit("lambda1", "Creates a new function with arg1 argument and given body",
            [lambda r, args: r.return_type if isinstance(r, FuncType) else Type.T],
            FuncType(AnyType.T, [Number.T]),
            lambda1,
            lambda r, args: FuncType(args[0], [Number.T]),
            ARGS_COMPILE_ONLY),
        Unit("lambda2", "Creates a new function with arg1 and arg2 arguments and given body",
            [lambda r, args: r.return_type if isinstance(r, FuncType) else Type.T],
            FuncType(AnyType.T, [Number.T, Number.T]),
            lambda2,
            lambda r, args: FuncType(args[0], [Number.T, Number.T]), ARGS_COMPILE_ONLY),

        Unit("combine", "Call the second function with the given arguments, pass the result to the first and return the result of its execution",
            [(lambda r, args: FuncType(r.return_type, [AnyType.T])),
             (lambda r, args: FuncType(args[0].argument_types[0], r.argument_types))],
             AnyFuncType.T, lambda f, g: (lambda *a: f(g(*a))),
             lambda r, args: FuncType(args[0].return_type, args[1].argument_types)),
        Unit("partial0", "Returns function with first argument of the given function with given value",
            [NotFuncType.T,
            (lambda r, args: FuncType(r.return_type, ([args[0]] + r.argument_types) if r.argument_types != AnyArgs.T else AnyArgs.T2))], 
            AnyFuncType.T, functools.partial(partial, 0),
            lambda r, args: FuncType(args[1].return_type, args[1].argument_types[1:])),
        Unit("partial1", "Returns function with first argument of the given function with given value",
            [NotFuncType.T,
            (lambda r, args: FuncType(r.return_type, (r.argument_types[:1] + [args[0]] + r.argument_types[1:]) if r.argument_types != AnyArgs.T else AnyArgs.T2))], 
            AnyFuncType.T, functools.partial(partial, 1),
            lambda r, args: FuncType(args[1].return_type, args[1].argument_types[:1] + args[1].argument_types[2:])),
        Unit("partial2", "Returns function with first argument of the given function with given value",
            [NotFuncType.T,
            (lambda r, args: FuncType(r.return_type, (r.argument_types[:2] + [args[0]] + r.argument_types[2:]) if r.argument_types != AnyArgs.T else AnyArgs.T3))], 
            AnyFuncType.T, functools.partial(partial, 2),
            lambda r, args: FuncType(args[1].return_type, args[1].argument_types[:2] + args[1].argument_types[3:])),

        Unit("array-new", "Returns new array of given type", 
            [lambda r, args: TypeType(r.underlying_type) if isinstance(r, Array) else TypeType.T], 
             Array(AnyType.T), lambda a: [], 
             lambda r, args: Array(args[0].base)),
        Unit("len", "Length of an array", [Array(Type.T)], Number.T, lambda a: len(a)),
        Unit("head", "Returns the first element of the array", 
            [lambda r, args: Array(r)], AnyType.T, head,
            lambda r, args: args[0].underlying_type),
        Unit("tail", "Returns the given array without the first element", 
            [lambda r, args: r if isinstance(r, Array) else Array(Type.T)], Array(AnyType.T), lambda a: a[1:],
            lambda r, args: args[0]),
        Unit("deref", "Returns element of an array with the given index", 
            [(lambda r, args: Array(r)), Number.T], 
            AnyType.T, lambda a, idx: deref(a, idx),
            lambda r, args: args[0].underlying_type),
        Unit("int-deref", "Returns element of an integer array with the given index", 
            [Array(Number.T), Number.T],
            Number.T, lambda a, idx: deref(a, idx)),
        Unit("slice", "Returns subset of the given array from given lo to hi",
             [lambda r, args: r if isinstance(r, Array) else Array(Type.T), Number.T, Number.T], Array(AnyType.T),
             slice_func, lambda r, args: args[0]),
        Unit("append", "Returns given array with appended element at the end",
            [lambda r, args: r if isinstance(r, Array) else Array(Type.T), lambda r, args: args[0].underlying_type], Array(AnyType.T),
            lambda arr, v: arr + [v],
            lambda r, args: args[0]),
        Unit("sort", "Returns given array sorted - each subsequent element is greater or equal",
            [lambda r, args: r if isinstance(r, Array) else Array(Type.T)], Array(AnyType.T),
            lambda arr: sorted(arr),
            lambda r, args: args[0]),
        Unit("reverse", "Returns array with values in reversed order",
            [lambda r, args: r if isinstance(r, Array) else Array(Type.T)], Array(AnyType.T),
            lambda arr: list(reversed(arr)),
            lambda r, args: args[0]),

        Unit("strlen", "Length of a string", [String.T], Number.T, lambda a: len(a)),
        Unit("str_index", "Returns a character from the given string at a given position", 
            [Number.T, String.T], String.T, lambda idx, a: a[int(idx)]),
        Unit("str_concat", "Returns a string that is achieved by writing the second input string after the first input string", 
            [String.T, String.T], String.T, lambda a, b: a + b),
        Unit("str_split", "Given a string and a delimiter, returns an array of strings consisting of substrings between the instances of the delimiters in the input string", 
            [String.T, String.T], Array(String.T), lambda s, delimiter: [x for x in s.split(delimiter) if x]),

        Unit("dict-new", "Creates new dictionary with give types of keys and values", 
            [ArgType(TypeType.T, lambda r, args: TypeType(r.key_type) if isinstance(r, Dict) else TypeType.T), 
             ArgType(TypeType.T, lambda r, args: TypeType(r.value_type) if isinstance(r, Dict) else TypeType.T)], 
            Dict(AnyType.T, AnyType.T),
            lambda key_type, value_type: {},
            lambda r, args: Dict(args[0].base, args[1].base)),
        Unit("dict-insert", "Inserts key value pair into dictionary", 
            [lambda r, args: r if isinstance(r, Dict) else Dict(Type.T, Type.T), 
            lambda r, args: args[0].key_type, lambda r, args: args[0].value_type], 
            Dict(AnyType.T, AnyType.T),
            dict_insert,
            lambda r, args: args[0]),
        Unit("dict-keys", "Returns list of keys from given dictionary",
            [lambda r, args: Dict(r.underlying_type, Type.T) if isinstance(r, Array) else Dict(Type.T, Type.T)], Array(AnyType.T),
            lambda d: list(sorted(d.keys())),
            lambda r, args: Array(args[0].key_type)),
    ]
    return {unit.name: unit for unit in units}


def program_to_unit(lisp_units, name, description, args, return_type, program):
    compute = compile_func(lisp_units, name, program, args, return_type)
    return Unit(name, description, [arg[1] for arg in args], return_type, compute)


def load_lisp_units():
    lisp_units = load_default_lisp_units()
    lisp_units['square'] = program_to_unit(lisp_units,
        'square', 'Computed the given number squared', 
        [('x', Number.T)], Number.T, ['*', 'x', 'x'])
    #lisp_units['is_even'] = program_to_unit(lisp_units,
    #    'is_even', 'Check if the given number is even - not divisible by two', 
    #    [('x', Number.T)], Boolean.T, ['==', '0', ['%', 'x', '2']])
    #lisp_units['is_odd'] = program_to_unit(lisp_units,
    #    'is_odd', 'Check if the given number is odd - divisible by two',
    #    [('x', Number.T)], Boolean.T, ['==', '1', ['%', 'x', '2']])
    lisp_units['is_prime'] = program_to_unit(lisp_units,
        'is_prime', 'Check if the given number is prime',
        [('x', Number.T)], Boolean.T, 
        ['if', ['<=', 'x', '1'], 'false', ['!', ['reduce', ['map', ['map', ['range', '2', ['+', '1', ['sqrt', 'x']]], ['partial0', 'x', '%']], ['partial0', '0', '==']], 'false' , '||']]])
    lisp_units['min'] = program_to_unit(lisp_units,
        'min', 'Compute the minimum of two numbers',
        [('x', Number.T), ('y', Number.T)], Number.T, ['if', ['<=', 'x', 'y'], 'x', 'y'])
    lisp_units['max'] = program_to_unit(lisp_units,
        'max', 'Compute the maximum of two numbers',
        [('x', Number.T), ('y', Number.T)], Number.T, ['if', ['<=', 'x', 'y'], 'y', 'x'])
    lisp_units['str_min'] = program_to_unit(lisp_units,
        'str_min', 'Compute the minimum of two strings',
        [('x', String.T), ('y', String.T)], String.T, ['if', ['<=', 'x', 'y'], 'x', 'y'])
    lisp_units['str_max'] = program_to_unit(lisp_units,
        'str_max', 'Compute the maximum of two strings',
        [('x', String.T), ('y', String.T)], String.T, ['if', ['<=', 'x', 'y'], 'y', 'x'])
    lisp_units['contains'] = program_to_unit(lisp_units,
       'contains', 'Check if a given element is contained in the given array',
       [('arr', Array(Number.T)), ('x', Number.T)], Boolean.T, 
       ['reduce', ['map', 'arr', ['partial0', 'x', '==']], 'false', '||'])
    lisp_units['is_sorted_helper'] = program_to_unit(lisp_units,
        'is_sorted_helper', 'Check if an array is sorted and the the given element is less than or equal to the first element of the array',
        [('x', Number.T), ('xs', Array(Number.T))], Boolean.T, 
        ['if', ['==', ['len', 'xs'], '0'], 'true', ['&&', ['<=', 'x', ['head', 'xs']], ['is_sorted_helper', ['head', 'xs'], ['tail', 'xs']]]])
    lisp_units['is_sorted'] = program_to_unit(lisp_units,
                                              'is_sorted', 'Check if the given array is sorted',
        [('arr', Array(Number.T))], Boolean.T, 
        ['if', ['==', ['len', 'arr'], '0'], 'true', ['is_sorted_helper', ['head', 'arr'], ['tail', 'arr']]])
    lisp_units['abs'] = program_to_unit(lisp_units,
        'abs', 'Absolute value of given integer',
        [('a', Number.T)], Number.T,
        ['if', ['>', 'a', '0'], 'a', ['-', '0', 'a']])
    lisp_units['digits'] = program_to_unit(lisp_units,
        'digits', 'Returns array of digits for given integer',
        [('a', Number.T)], Array(Number.T),
        ["if",
            ["<", ["abs", "a"], "10"], 
            ["append", ["array-new", "int"], ["abs", "a"]], 
            ["append", ["digits", ["/", ["abs", "a"], "10"]], ["%", ["abs", "a"], "10"]]]
    )
    return lisp_units


def load_specification_units():
    lisp_units = load_lisp_units()
    def crossproduct(left, right):
        res = []
        for x in left:
            x = [x] if not isinstance(x, tuple) else list(x)
            for y in right:
                y = [y] if not isinstance(y, tuple) else list(y)
                res.append(tuple(x + y))
        return res
    TOTAL_CHOICES_EVALUATE = 10000
    def choice(argument_map, context, args, generator, cond, ret):
        arguments = []
        if isinstance(args[0], list):
            args = args[0][1:]
        else:
            args = [args[0]]
        gen_f = compile_statement(
            lisp_units, generator[0], argument_map=argument_map, context=context)
        argument_map_ = {k: v for k, v in argument_map.items()}
        for arg in args:
            argument_map_[arg] = Number.T
        cond_f = compile_statement(
            lisp_units, cond[0], argument_map_, context
        )
        ret_f = compile_statement(
            lisp_units, ret[0], argument_map_, context
        )
        result = []
        count = 0
        for items in gen_f():
            context.enter_scope(closure=True)
            if isinstance(items, tuple):
                for arg, item in zip(args, items):
                    context[arg] = item
            else:
                context[args[0]] = items
            if cond_f():
                result.append(ret_f())
            context.exit_scope()
            count += 1
            if count > TOTAL_CHOICES_EVALUATE:
                break
        return result
    lisp_units['crossproduct'] = Unit('crossproduct', '', [AnyType.T, AnyType.T], AnyType.T, crossproduct)
    lisp_units['choice'] = Unit('choice', '', [AnyType.T, AnyType.T, AnyType.T, AnyType.T], AnyType.T, choice, arguments_pass_type=ARGS_PASS_LISP)
    lisp_units['str'] = Unit('str', '', [AnyType.T], String.T, lambda x: str(x))
    return lisp_units


_EXECUTION_CACHE = {}


def lists_to_tuples(statement):
    if isinstance(statement, list):
        return tuple(lists_to_tuples(x) for x in statement)
    elif isinstance(statement, dict):
        keys = tuple(sorted(statement.keys()))
        return (keys, tuple(statement[x] for x in keys))
    elif hasattr(statement, '_original_lisp_code'):
        return lists_to_tuples(statement._original_lisp_code)
    else:
        return statement


def compile_statement(lisp_units, short_tree, argument_map=None, context=None, trace=None):
    global _EXECUTION_CACHE

    def _compile_statement(statement):
        if isinstance(statement, six.string_types):
            unit = lisp_units.get(statement, None)
            if unit:
                assert unit.compute is not None, "Unit %s doesn't have compute code." % unit.name
                unit.compute._original_lisp_code = statement
                return unit.compute
            elif argument_map and statement in argument_map:
                ret = lambda: context[statement]
                ret._original_lisp_code = statement
                return ret
            else:
                raise ValueError("Unknown symbol %s, arguments: %s" % (statement, argument_map))
        elif isinstance(statement, (int, float)):
            return lambda: statement
        elif isinstance(statement, list):
            func, args = statement[0], statement[1:]
            unit = lisp_units.get(func, None)
            arguments_pass_type = False
            if unit:
                arguments_pass_type = unit.arguments_pass_type
                if len(unit.args) != len(args):
                  raise ValueError("Wrong number of arguments for %s: %s vs %s" % (unit.name, unit.args, args))
            elif unit is None and argument_map and func in argument_map:
                pass
            else:
                raise ValueError("Unknown function %s, arguments: %s" % (func, argument_map))

            values = []
            for arg in args:
                call = (isinstance(arg, (list, int, float)) or 
                    (argument_map and arg in argument_map) or 
                    (arg in lisp_units and not lisp_units[arg].args))
                if arguments_pass_type != ARGS_PASS_LISP:
                    if argument_map is not None:
                        lambda_arg1, lambda_arg2 = 'arg1' in argument_map, 'arg2' in argument_map
                        if unit and unit.name == 'lambda1':
                            argument_map['arg1'] = Number.T
                            argument_map['self'] = FuncType(AnyType.T, [Number.T])
                        if unit and unit.name == 'lambda2':
                            argument_map['arg1'] = Number.T
                            argument_map['arg2'] = Number.T
                            argument_map['self'] = FuncType(AnyType.T, [Number.T, Number.T])

                    # Call the function.
                    values.append((_compile_statement(arg), call))

                    if argument_map:
                        if unit and unit.name in ('lambda1', 'lambda2') and 'arg1' in argument_map and not lambda_arg1:
                            argument_map.pop('arg1')
                        if unit and unit.name == 'lambda2' and 'arg2' in argument_map and not lambda_arg2:
                            argument_map.pop('arg2')
                else:
                    values.append((arg, call))

            def call_unit():
                try:
                    use_cache = True
                    if arguments_pass_type in [ARGS_COMPILE_ONLY, ARGS_PASS_LISP]:
                        arg_values = values
                        use_cache = False
                    else:
                        arg_values = [value() if call else value for value, call in values]

                    if func == 'dict-new':
                        use_cache = False

                    if trace is not None:
                        trace_id = trace.add_call(func, arg_values)

                    if use_cache:
                        try:
                            cache_key1 = lists_to_tuples(statement)
                        except Exception as e:
                            use_cache = False
                        try:
                            cache_key2 = lists_to_tuples(arg_values)
                        except Exception as e:
                            use_cache = False
                        try:
                            cache_key3 = lists_to_tuples(argument_map)
                        except Exception as e:
                            use_cache = False

                        if use_cache:
                            cache_key = (cache_key1, cache_key2, cache_key3)

                            if cache_key in _EXECUTION_CACHE:
                                ret = _EXECUTION_CACHE[cache_key]

                                if trace is not None:
                                    trace.add_result(trace_id, ret)

                                return ret

                    if argument_map and func in argument_map:
                        ret = context[func](*arg_values)
                    else:
                        if arguments_pass_type == ARGS_COMPILE_ONLY:
                            ret = unit.compute(context, *arg_values)
                        elif arguments_pass_type == ARGS_PASS_LISP:
                            ret = unit.compute(argument_map, context, *arg_values)
                        else:
                            ret = unit.compute(*arg_values)

                    if use_cache:
                        _EXECUTION_CACHE[cache_key] = ret

                    if trace is not None:
                        trace.add_result(trace_id, ret)

                    return ret
                    
                except TypeError as e:
                    print("TypeError %s calling %s, arguments: %s. Program: %s" % (
                        str(e), func, values, statement))
                    raise
            ret = call_unit
            ret._original_lisp_code = statement
            return ret
        else:
            raise ValueError("Unexpected type of statement: %s" % str(statement))


    return _compile_statement(short_tree)


class ScopeContext(object):

    def __init__(self):
        self.stack = []

    def enter_scope(self, closure=False):
        if closure and len(self.stack) > 0:
            self.stack.append(copy.copy(self.stack[-1]))
        else:
            self.stack.append({})

    def exit_scope(self):
        self.stack.pop()

    def __setitem__(self, key, value):
        self.stack[-1][key] = value

    def __getitem__(self, key):
        return self.stack[-1][key]

    def pop(self, key):
        self.stack[-1].pop(key)


def compile_func(lisp_units, name, statement, arguments, return_type, trace=None):
    """Compiles lisp statement into python function.

    Args:
        lisp_units: available units.
        name: name of the function.
        statement: lisp code.
        arguments: list of tuples (name, type), argument names and types.
        return_type: resulting type.
        trace: Trace object or None.
    """
    context = ScopeContext()
    argument_map = dict(arguments) if arguments else {}
    argument_map[name] = FuncType(return_type, arguments)

    def final_func(func, *args):
        """Resulting function after compilation.

        Args:
            func: python func to call (compiled from lisp).
            args: list of values as argumnets of the function func.
        """
        context.enter_scope()
        context[name] = functools.partial(final_func, func)
        for arg, value in zip(arguments, args):
            context[arg[0]] = value
        # TODO: check all args are filled in.
        res = func()
        context.exit_scope()
        return res

    return functools.partial(
        final_func, compile_statement(lisp_units, statement, argument_map, context, trace=trace))


def test_lisp_validity(lisp_units, short_tree, args_map, root_type, constants = {}, name=None):
    #print(short_tree, root_type)
    if isinstance(short_tree, six.string_types):
        if short_tree in lisp_units:
            x = lisp_units[short_tree]
            r = x.return_type
            if x.args:
                if not isinstance(root_type, FuncType):
                    raise TypeError("Using function %s when %s expected." % (x, root_type))
                if any_type(root_type.return_type) and not root_type.return_type.compatible(root_type.return_type):
                    raise TypeError("Function %s return type %s is not compatible with expected %s return type %s" % (
                        x, x.return_type, root_type, root_type.return_type))
                if len(x.args) != len(root_type.argument_types):
                    raise TypeError("Function %s number of arguments %s doesn't match %s." % (
                        x, x.args, root_type))
                args = []
                for i, arg in enumerate(x.args):
                    if callable(arg):
                        tp = arg(x.return_type, root_type.argument_types[:i])
                    else:
                        tp = arg
                    if not any_type(root_type.argument_types[i]) and not root_type.argument_types[i].compatible(tp):
                        raise TypeError("Passing function %s(%s) when %s expected. Argument %d mismatch: %s vs %s." % (
                            x, [arg for arg in x.args], root_type, i, tp, root_type.argument_types[i]))
                #FuncType(x.return_type, [arg for arg in root_type.args])
                r = FuncType(x.return_type if not any_type(x.return_type) else root_type.return_type,
                    [arg if not callable(arg) else root_arg for arg, root_arg in zip(x.args, root_type.argument_types)])                
            # ????
            # if x.computed_return_type is not None:
            #     print(x, x.computed_return_type, r, root_type)
            #     r = functools.partial(x.computed_return_type, root_type)
            #print(x.name, r, x.computed_return_type, root_type)
        elif short_tree in constants:
            x = constants[short_tree]
            r = x.get_type()
        elif short_tree in args_map:
            r = args_map[short_tree]
        else:
            raise ValueError("Unknown symbol %s. Current args: %s" % (
                short_tree, args_map))
        if not r.compatible(root_type):
            raise TypeError(
                "Symbol %s with return type %s is not compatible with "
                "expected type %s" % (short_tree, r, root_type))
    else:
        if short_tree[0] in constants and short_tree[0] not in mapping:
            # TODO: validate built-in functions
            return
        if short_tree[0] in args_map:
            r = args_map[short_tree[0]]
            return r

        x = lisp_units[short_tree[0]]
        r = x.return_type
        args = []
        if not r.compatible(root_type):
            raise TypeError("Return type %s of function %s doesn't match with expected return type %s: %s" % (
                r, short_tree[0], root_type, short_tree))
        assert r.compatible(root_type), (short_tree, r, root_type)

        if len(x.args) + 1 != len(short_tree):
            raise ValueError("Wrong number of arguments in %s, expected: %s" % (
                short_tree, x.args))
        assert len(x.args) + 1 == len(short_tree), short_tree

        # Handling lambdas.
        if x.name == 'lambda1':
            assert len(short_tree) == 2
            new_args_map = copy.copy(args_map)
            new_args_map['arg1'] = Number.T
            new_args_map['self'] = FuncType(root_type, [Number.T])
            ret = test_lisp_validity(lisp_units, short_tree[1], new_args_map, root_type.return_type, constants)
            r = FuncType(ret, [Number.T])
        elif x.name == 'lambda2':
            assert len(short_tree) == 2
            new_args_map = copy.copy(args_map)
            new_args_map['arg1'] = Number.T
            new_args_map['arg2'] = Number.T
            new_args_map['self'] = FuncType(root_type, [Number.T, Number.T])
            ret = test_lisp_validity(lisp_units, short_tree[1], new_args_map, root_type.return_type, constants)
            r = FuncType(ret, [Number.T, Number.T])
        else:
            # Regular function call.

            for arg, sub in zip(x.args, short_tree[1:]):
                if callable(arg):
                    try:
                        tp = arg(root_type, args)
                    except:
                        print("Error while executing: %s, arg: %s, root_type: %s, args: %s" % (
                            short_tree, arg, root_type, args))
                        raise
                else:
                    tp = arg
                # print(sub, arg, tp, root_type, args, lisp_units[short_tree[0]].args)
                args.append(test_lisp_validity(lisp_units, sub, args_map, tp, constants))
                if args[-1].__class__ ==  Type:
                    raise TypeError("Argument %s resolved to %s." % (sub, args[-1]))

            if x.computed_return_type is not None:
                r = x.computed_return_type(root_type, args)
                assert r.compatible(root_type), "Computed type for %s -> %s doesn't match expected type %s. Probably issue with argument computation." % (short_tree, r, root_type)

    # print(short_tree, r)
    return r


def process_code_tree(code_tree, mapping):
    def dfs(node):
        if isinstance(node, six.string_types):
            return mapping.get(node, node)
        elif isinstance(node, list):
            return [mapping.get(node[0], node[0])] + [dfs(n) for n in node[1:]]
        else:
            assert False
    return dfs(code_tree)
