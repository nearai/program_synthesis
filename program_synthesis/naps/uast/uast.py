from __future__ import print_function

import functools
import six
import sys
import math
import numpy as np
import re
from operator import mul

from sortedcontainers import SortedDict, SortedSet

from .uast_watcher import WatcherEvent, tuplify

DEBUG_INFO = False

LARGEST_INT = 2 ** 64

OBJECT = "object"
BOOL = "bool"
CHAR = "char"
STRING = "char*"
INT = "int"
REAL = "real"
VOID = "void"
FUNC = "func"

if not six.PY2:
    long = int


def watchable(event_type):
    def watchable_internal(some_func):
        def wrapper(executor, context, *args, **kwargs):
            if not executor.watchers: # don't waste precious cycles if there are no watchers
                return some_func(executor, context, *args, **kwargs)

            assert len(kwargs) <= 1, "%s for %s" % (kwargs, some_func)
            all_args = list(args) + list(kwargs.values())

            executor._watch(WatcherEvent("before_" + event_type, executor, context, *all_args))
            ret = some_func(executor, context, *args, **kwargs)
            executor._watch(WatcherEvent("after_" + event_type, executor, context, ret, *all_args))

            return ret

        return wrapper

    return watchable_internal


class IO_t:
    SCANNER = 'scanner'
    PRINTER = 'printer'

    next_int = ['invoke', INT, '_io_next_int', []]
    next_real = ['invoke', INT, '_io_next_real', []]
    next_line = ['invoke', STRING, '_io_next_line', []]
    next_string = ['invoke', STRING, '_io_next_word', []]
    def print_(self, x):
        return ['invoke', VOID, '_io_print', [x]]

    def println(self, x):
        return ['invoke', VOID, '_io_println', [x]]

    def __init__(self):
        self.func_to_type = {}
        self.func_to_type[self.next_int[2]] = INT
        self.func_to_type[self.next_real[2]] = REAL
        self.func_to_type[self.next_line[2]] = STRING
        self.func_to_type[self.next_string[2]] = STRING


IO = IO_t()


GLOBALS_NAME = "__globals__"


class UASTNotImplementedException(Exception):
    def __init__(self, feature, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.feature = feature

    def __str__(self):
        return "UAST Not Implemented: %s" % self.feature


class UASTTimeLimitExceeded(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class UASTParseError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def var(name, type_):
    return ["var", type_, name]


def get_expr_type(var):
    return var[1]


def get_var_name(var):
    assert var[0] == 'var'
    return var[2]


def set_var_name(var, name):
    var[2] = name


def constant(type_, value):
    return ["val", type_, value]


def func(name, return_type=VOID):
    return ["func", return_type, name, [], [], []]


def get_func_return_type(f):
    assert f[0] in ['func', 'ctor'], f[0]
    return f[1]


def get_func_name(f):
    assert f[0] in ['func', 'ctor'], f[0]
    return f[2]


def set_func_name(f, new_name):
    assert f[0] in ['func', 'ctor'], f[0]
    f[2] = new_name


def get_func_args(f):
    assert f[0] in ['func', 'ctor'], f[0]
    return f[3]


def get_func_vars(f):
    assert f[0] in ['func', 'ctor'], f[0]
    return f[4]


def get_func_body(f):
    assert f[0] in ['func', 'ctor'], f[0]
    return f[5]


def record(name):
    return ["record", name, {}]


def get_record_name(record):
    return record[1]


def get_record_fields(record):
    return record[2]


def func_call(func_name, args, type_):
    return ["invoke", type_, func_name, args]


def assign(lhs, rhs):
    return ["assign", rhs[1], lhs, rhs]


def field(jcontext, obj, field):
    type_ = get_expr_type(obj)
    assert isinstance(type_, six.string_types), type_
    assert type_[-1] == '#', type_
    record = jcontext.get_record(type_[:-1])    
    return ["field", get_expr_type(get_record_fields(record)[field]), obj, field]


def type_array(subtype):
    return subtype + "*"


def type_set(subtype):
    return subtype + "%"


def type_map(subtype1, subtype2):
    return '<' + subtype1 + '|' + subtype2 + ">"


def type_record(name):
    return name + "#"


def get_array_subtype(tp):
    assert tp[-1] == '*', tp
    return tp[:-1]


def get_map_key_type(tp):
    assert tp[0] == '<', tp
    assert tp[-1] == '>'
    ret = ""
    balance = 0
    for ch in tp[1:-1]:
        if ch == '<':
            balance += 1
        elif ch == '>':
            assert balance > 0
            balance -= 1
        elif ch == '|' and balance == 0:
            break
        ret += ch
    return ret


def get_map_value_type(tp):
    assert tp[0] == '<', tp
    assert tp[-1] == '>'
    ret = ""
    balance = 0
    saw_pipe = False
    for ch in tp[1:-1]:
        if saw_pipe:
            ret += ch
        if ch == '<':
            balance += 1
        elif ch == '>':
            assert balance > 0
            balance -= 1
        elif ch == '>':
            assert saw_pipe
            break
        elif ch == '|' and balance == 0:
            saw_pipe = True
    return ret


def type_to_record_name(tp):
    assert tp[-1] == '#', "%s <> %s" % (tp, tp[-1])
    return tp[:-1]


def is_array(tp):
    return tp[-1] == '*'


def is_record_type(tp):
    return tp[-1] in ['#']


def is_int_type(tp):  # doesn't include char!
    return tp in [INT]


def is_set_type(tp):
    return tp[-1] in ['%']


def is_map_type(tp):
    return tp[-1] in ['>']


def if_(cond, then, else_):
    return ["if", VOID, cond, then, else_]


def ternary(cond, then, else_):
    return ["?:", arithmetic_op_type(get_expr_type(then), get_expr_type(else_), allow_same=True), cond, then, else_]


def while_(cond, body, finally_):
    return ["while", VOID, cond, body, finally_]


def for_each(var, collection, body):
    return ["foreach", VOID, var, collection, body]


def arithmetic_op_type(tp1, tp2, allow_same=False):
    if allow_same and (tp1 == tp2 or tp1 == OBJECT or tp2 == OBJECT): # TODO: check that we are not comparing object and value type
        return tp1

    for pr in [REAL, INT, CHAR, BOOL]:
        if tp1 == pr or tp2 == pr:
            return pr
    raise UASTNotImplementedException("Arithmetic op on %s and %s" % (tp1, tp2))


def convert_binary_expression(arg1, arg2, operator):
    if get_expr_type(arg1) == BOOL and get_expr_type(arg2) == BOOL \
            and operator in ['|', '&', '^']:
        operator += operator

    if operator in ['&&', '||', '==', '!=', '<', '<=', '>', '>=', '^^']:
        return func_call(operator if operator != '^^' else '^', [arg1, arg2], BOOL)

    if get_expr_type(arg1) == STRING or get_expr_type(arg2) == STRING:
        if get_expr_type(arg1) in [STRING, CHAR] and get_expr_type(arg2) in [STRING, CHAR]:
            assert operator == '+', operator
            return func_call('concat', [arg1, arg2], STRING)

        elif get_expr_type(arg1) in [STRING, VOID] and get_expr_type(arg2) in [STRING, VOID]:
            assert operator in ['==', '!='], operator
            return func_call(operator, [arg1, arg2], BOOL)

        elif get_expr_type(arg1) == STRING:
            assert operator == '+', operator
            return func_call('concat', [arg1, func_call('str', [arg2], STRING)], STRING)
        elif get_expr_type(arg2) == STRING:
            assert operator == '+', operator
            return func_call('concat', [func_call('str', [arg1], STRING), arg2], STRING)
        assert False, "%s %s %s" % (get_expr_type(arg1), operator, get_expr_type(arg2))

    if operator in ['+', '*', '%', '&', '|', '^', '-', '/', '>>', '<<']:
        tp_ = arithmetic_op_type(get_expr_type(arg1), get_expr_type(arg2))
        return func_call(operator, [arg1, arg2], tp_)
    else:
        raise UASTNotImplementedException("operator %s" % operator)


def is_assigneable(expr):
    return expr[0] in ['var', 'field'] or expr[0] == 'invoke' and expr[2] == 'array_index'


def assert_val_matches_type(val, tp):
    if tp == '?':
        return
    if not val_matches_type(val, tp):
        if isinstance(val, float) and is_int_type(tp):
            raise UASTNotImplementedException("Implicit cast from REAL to INT")
        if val is None and is_int_type(tp):
            raise UASTNotImplementedException("Implicit cast from NULL to INT")
        assert False, "Type mismatch.\n    Type: %s;\n    Val: %s\n    Val type: %s\n" % (tp, val, type(val))


def val_matches_type(val, tp, verbose=False):
    if is_int_type(tp) or tp == CHAR:
        # allow implicit conversion from float to int
        return isinstance(val, float) or isinstance(val, int) or isinstance(val, long)
    elif tp in [REAL]:
        return isinstance(val, float) or isinstance(val, int)
    elif tp in [STRING]:
        return isinstance(val, six.string_types) or val is None
    elif tp in [BOOL]:
        return isinstance(val, bool)
    elif tp[-1] in ["*"]:
        return isinstance(val, list) or val is None
    elif tp[-1] in ['#']:
        return isinstance(val, dict) or val is None
    elif tp[-1] in ['>']:
        return isinstance(val, SortedDict) or val is None
    elif tp[-1] in ['%']:
        return isinstance(val, SortedSet) or val is None
    elif tp == 'void':
        return val is None
    elif tp == 'func':
        return isinstance(val, six.string_types)
    elif tp in 'object':
        return not isinstance(val, int) and not isinstance(val, long) and not isinstance(val, float) and not isinstance(val, bool)
    elif tp in [IO.SCANNER, IO.PRINTER]:
        return val is None
    else:
        assert False, tp


def can_cast(to_, from_):
    if from_ == '?':
        return True
    if (to_[-1] in ['*', '#', '>', '%'] or to_ == OBJECT) and \
            (from_[-1] in ['*', '#', '>', '%'] or from_ == OBJECT):
        return True
    return to_ in [INT, REAL, CHAR] and from_ in [INT, REAL, CHAR, STRING]


def get_block_statements(block):
    assert isinstance(block, list)
    return block


def default_value(ret_type):
    if ret_type in [INT, REAL, CHAR]:
        return 0
    elif ret_type in [STRING]:
        return ""
    elif ret_type == BOOL:
        return False
    return None


# parse_context is either JContext or CContext
def prepare_global_var_and_func(parse_context):
    gi_fname = GLOBALS_NAME + ".__init__"

    globals_ = record(GLOBALS_NAME)
    parse_context.register_type(GLOBALS_NAME, type_record(GLOBALS_NAME))
    parse_context.program['types'].append(globals_)

    parse_context.globals_record = globals_
    parse_context.globals_init_var = var(GLOBALS_NAME, type_record(GLOBALS_NAME))
    parse_context.globals_init_func = func(gi_fname, VOID)

    return gi_fname


class InputSchemaExtractorNotSupportedException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class InputSchemaExtractor(object):
    def __init__(self, data, attempt_multi_test=False):
        super(InputSchemaExtractor, self).__init__()
        self.uast = data

        self.is_multi_test = attempt_multi_test

        self.multi_test_var = None
        self.multi_test_iter = None
        self.multi_test_loop = None
        self.inside_multi_test_loop = False

        self.funcs = {get_func_name(func): func for func in data['funcs']}
        self.types = {get_record_name(record): record for record in data['types']}
        self.schema = []
        self.cur_schema = self.schema
        self.var_map = {}
        self.arr_map = {}
        self.init_vals = {}

        self.var_map_assigns = {}
        self.arr_map_assigns = {}
        self.arr_map_inits = {}
        self.bypassed_arrays = set()
        self.remove_vars = set()

        self.not_impl_stack = []
        self.loop_stack = []
        self.func_stack = []

        self.cur_branch_out_len = 0
        self.max_branch_out_len = 0
        self.output_type = None

        self.funcs_visited = {}
        self.funcs_with_reads = set()
        self.funcs_with_writes = set()
        self.func_returns = {}

        self.num_args = 0

        # datastructures to postprocess code
        self.replace_with_noops = []
        self.process_input_blocks = []
        self.process_output_blocks = []
        self.process_invokes_with_reads = []
        self.process_invokes_with_writes = []


    def push_not_impl(self, s):
        self.not_impl_stack.append(s)


    def pop_not_impl(self):
        self.not_impl_stack.pop()


    def check_not_impl(self):
        if self.not_impl_stack:
            raise InputSchemaExtractorNotSupportedException(self.not_impl_stack[-1])


    def next_arg_name(self):
        if self.num_args >= 26:
            raise InputSchemaExtractorNotSupportedException("More than 26 arguments")
        self.num_args += 1
        return chr(ord('a') + self.num_args - 1)


    def crawl_stmt(self, stmt):
        # TODO: presently not supporting reading lines
        def hack_fix_me(s):
            if s == 'line': return 'word'
            return s

        if stmt[0] == 'if': # TODO: properly handle this
            self.push_not_impl("IO inside if")
            self.crawl_stmt(stmt[2])

            out_depth_before = self.cur_branch_out_len
            self.crawl_stmt_list(stmt[3])

            out_depth_after = self.cur_branch_out_len
            self.cur_branch_out_len = out_depth_before

            self.crawl_stmt_list(stmt[4])

            self.cur_branch_out_len = max(self.cur_branch_out_len, out_depth_after)

            self.pop_not_impl()

        elif stmt[0] == 'foreach':
            self.push_not_impl("IO inside foreach")
            self.loop_stack.append(None)
            self.crawl_stmt_list(stmt[4])
            self.loop_stack.pop()
            self.pop_not_impl()

        elif stmt[0] == 'while':
            cond = stmt[2]
            body = stmt[3]

            is_up_to_t = self.is_multi_test and self.multi_test_var is not None
            if is_up_to_t:
                is_up_to_t = cond[0] == 'invoke' and cond[2] in ('<', '<=') and cond[3][1][0] == 'var' and cond[3][1][2] == self.multi_test_var[0][2] and cond[3][0][0] == 'var' and cond[3][0][2] in self.init_vals
                init_val = self.init_vals[cond[3][0][2]] if is_up_to_t else None
                is_up_to_t = is_up_to_t and init_val is not None and (cond[2] == '<' and init_val == 0) #or cond[2] == '<=' and init_val == 1) # TODO: add support for 1-based indexing. Presently turned off because we use the variable to index into the tests
                if is_up_to_t:
                    self.multi_test_iter = cond[3][0]
                    self.multi_test_loop = stmt
                    assert len(self.schema) == 1 and self.schema[0] == self.multi_test_var[1], "%s <> %s" % (self.schema, self.multi_test_var[1])
                    self.process_input_blocks.append([self.schema[0], cond[3][1]])
                #print(('T', is_up_to_t, cond, init_val, cond[3][0][2], self.multi_test_var))

                else: # try while(t-->0)
                    if cond[0] == 'invoke' and cond[2] in ('>', '!='):
                        #print("Step1...")
                        cond_lhs = cond[3][0]
                        cond_rhs = cond[3][1]
                        common_cond = cond_rhs[0] == 'val' and cond_rhs[2] == 0
                        while_t_minus_minus = common_cond and cond_lhs[0] == 'invoke' and cond_lhs[2] == '+' and cond_lhs[3][1][0] == 'val' and cond_lhs[3][1][2] == 1
                        if while_t_minus_minus:  
                            #print("Step2...")
                            while_t_minus_minus = cond_lhs[3][0][0] == 'assign' and cond_lhs[3][0][2][0] == 'var' and cond_lhs[3][0][2][2] == self.multi_test_var[0][2]

                        while_t = False
                        if not while_t_minus_minus:
                            while_t = common_cond and cond_lhs[0] == 'var' and cond_lhs[2] == self.multi_test_var[0][2]
                            while_t = while_t and body and body[-1][0] == 'assign' and body[-1][2][0] == 'var' and body[-1][2][2] == cond_lhs[2]
                            if while_t:
                                assign_rhs = body[-1][3]
                                #print("ASSIGN_RHS", assign_rhs)
                                while_t = assign_rhs[0] == 'invoke' and assign_rhs[2] == '-' and assign_rhs[3][0][0] == 'var' and assign_rhs[3][0][2] == cond_lhs[2]
                            #print(body[-1])

                        if while_t_minus_minus or while_t:
                            if while_t:
                                body.pop()
                                if body and body[-1][0] == 'var' and body[-1][2] == self.multi_test_var[0][2]:
                                    body.pop()

                            # TODO: would make sense to check if assign is correct, but probabilisticly it probably is :)
                            #print("Step3...")
                            self.multi_test_iter = ["var", INT, "ti"] #TODO: ti should be available
                            self.multi_test_loop = stmt
                            is_up_to_t = True
                            assert len(self.schema) == 1 and self.schema[0] == self.multi_test_var[1], "%s <> %s" % (self.schema, self.multi_test_var[1])

                            new_lhs = self.multi_test_iter
                            new_rhs = cond_lhs[3][0][2] if while_t_minus_minus else cond_lhs
                            stmt[2] = cond = ["invoke", BOOL, "<", [new_lhs, new_rhs]]
                            stmt[4] = [["assign", INT, self.multi_test_iter, ['invoke', INT, '+', [self.multi_test_iter, ['val', INT, 1]]]]]
                            self.process_input_blocks.append([self.schema[0], new_rhs])
                                



            is_up_to_n = cond[0] == 'invoke' and cond[2] in ('<', '<=') and cond[3][1][0] == 'var' and cond[3][1][2] in self.var_map and cond[3][0][0] == 'var' and cond[3][0][2] in self.init_vals

            init_val = self.init_vals[cond[3][0][2]] if is_up_to_n else None
            is_up_to_n = is_up_to_n and init_val is not None and (cond[2] == '<' and init_val == 0 or cond[2] == '<=' and init_val == 1)

            assert not is_up_to_n or not is_up_to_t

            #print(('N', is_up_to_n, cond, init_val, cond[3][0][2]))
            if is_up_to_t:
                if self.inside_multi_test_loop:
                    raise InputSchemaExtractorNotSupportedException("Iterating over the `t` inside iterating over `t` for multitest case")
                self.inside_multi_test_loop = True
                self.crawl_stmt_list(body)
                self.inside_multi_test_loop = False

            elif is_up_to_n:
                self.loop_stack.append(self.var_map[cond[3][1][2]])

                old_cur_schema = self.cur_schema
                self.cur_schema.append(['loop', VOID, []])
                self.cur_schema = self.schema[-1][2]

                self.crawl_stmt_list(body)

                if not self.cur_schema:
                    old_cur_schema.pop()
                self.cur_schema = old_cur_schema

                self.loop_stack.pop()

            else:
                self.push_not_impl("IO inside for other than range for on an input")
                self.loop_stack.append(None)
                self.crawl_stmt_list(body)
                self.loop_stack.pop()
                self.pop_not_impl()

        elif stmt[0] in ['break', 'continue', 'noop']:
            pass

        elif stmt[0] == 'return':
            func_name = get_func_name(self.func_stack[-1])
            if func_name not in self.func_returns:
                self.func_returns[func_name] = []
            self.func_returns[func_name].append(stmt)

            self.crawl_stmt(stmt[2])

        # Expressions
        elif stmt[0] == 'assign':
            if stmt[2][1] in [IO.SCANNER, IO.PRINTER]:
                self.replace_with_noops.append(stmt)
            else:
                ret = self.crawl_stmt(stmt[3])
                if ret is not None and stmt[2][0] == 'var':
                    if self.is_multi_test and self.multi_test_var is None:
                        self.multi_test_var = (stmt[2], ret)
                        self.replace_with_noops.append(stmt)
                    else:
                        self.var_map[stmt[2][2]] = ret
                        if stmt[3][0] != 'var':
                            self.var_map_assigns[stmt[2][2]] = stmt
                if ret is not None and stmt[2][0] == 'invoke' and stmt[2][2] == 'array_index' and stmt[2][3][0][0] == 'var' and stmt[2][3][1][0] == 'var':
                    self.arr_map[stmt[2][3][0][2]] = ret
                    self.arr_map_assigns[stmt[2][3][0][2]] = stmt
                    

                if stmt[3][0] == 'val' and stmt[2][0] == 'var':
                    #print("Assigning %s to %s" % (stmt[2][2], stmt[3][2]))
                    self.init_vals[stmt[2][2]] = stmt[3][2]
                if stmt[2][0] == 'var':
                    if stmt[2][2] not in self.arr_map_inits:
                        self.arr_map_inits[stmt[2][2]] = stmt
                    else:
                        self.arr_map_inits[stmt[2][2]] = False

        elif stmt[0] == 'var':
            if stmt[2] in self.var_map:
                return self.var_map[stmt[2]]

        elif stmt[0] == 'field':
            self.crawl_stmt(stmt[2])

        elif stmt[0] == 'val':
            pass

        elif stmt[0] == 'invoke':
            if stmt[2].startswith('_io_next_'):
                if self.is_multi_test and self.multi_test_var is not None and not self.inside_multi_test_loop:
                    raise InputSchemaExtractorNotSupportedException("Multitest schema with input outside of multitest while loop: %s" % stmt)

                self.funcs_with_reads.add(get_func_name(self.func_stack[-1]))

                self.check_not_impl()
                if len(self.loop_stack) > 1:
                    raise InputSchemaExtractorNotSupportedException("Nested loops")

                if not self.loop_stack:
                    new_entry = ['in', IO.func_to_type[stmt[2]], self.next_arg_name(), hack_fix_me(stmt[2].split('_')[-1])]
                else:
                    new_entry = ['in', type_array(IO.func_to_type[stmt[2]]), self.next_arg_name(), stmt[2].split('_')[-1]]
                    if self.loop_stack[-1][0] == 'in':
                        self.loop_stack[-1][0] = 'size'
                        self.loop_stack[-1][1] = INT
                        self.loop_stack[-1][2] = [new_entry[2]]
                    else:
                        assert self.loop_stack[-1][0] == 'size'
                        if new_entry[2] not in self.loop_stack[-1][2]:
                            self.loop_stack[-1][2].append(new_entry[2])

                self.process_input_blocks.append([new_entry, stmt])
                self.cur_schema.append(new_entry)
                return new_entry

            elif stmt[2].startswith('_io_print'):
                self.funcs_with_writes.add(get_func_name(self.func_stack[-1]))

                assert len(stmt[3]) in [0, 1], stmt
                if len(stmt[3]):
                    #if self.is_multi_test and not self.inside_multi_test_loop:
                    #    raise InputSchemaExtractorNotSupportedException("Multitest schema with output outside of multitest while loop")

                    if self.loop_stack or self.inside_multi_test_loop:
                        self.cur_branch_out_len = 2 # >1 means return a list
                    else:
                        self.cur_branch_out_len += 1

                    self.max_branch_out_len = max(self.max_branch_out_len, self.cur_branch_out_len)

                    new_output_type = get_expr_type(stmt[3][0])
                    if self.output_type is not None and self.output_type != new_output_type:
                        if self.output_type == 'char*' and not new_output_type.endswith('*'):
                            pass
                        elif not self.output_type.endswith('*') and new_output_type == 'char*':
                            self.output_type = 'char*'
                        else:
                            raise InputSchemaExtractorNotSupportedException("Mixing different output types: %s and %s" % (self.output_type, new_output_type))

                    else:
                        self.output_type = new_output_type

                    self.process_output_blocks.append(stmt)
                else:
                    self.replace_with_noops.append(stmt)

            else:
                assert not stmt[2].startswith('_io_')
                # TODO: invoke the function if it's a user-defined function
                for arg in stmt[3]:
                    if len(arg) <= 1:
                        assert False, "argument doesn't have two elements. Stmt: %s; arg: %s" % (stmt, arg)
                    if arg[1] in [IO.PRINTER, IO.SCANNER]:
                        arg[:] = ['val', VOID, None]
                    self.crawl_stmt(arg)
                if stmt[2] in self.funcs:
                    snapshot_var_map = self.var_map
                    self.var_map = {}

                    assert get_func_name(self.funcs[stmt[2]]) == stmt[2], "%s <> %s" % (self.funcs[stmt[2]], stmt[2])
                    self.crawl_func(self.funcs[stmt[2]])

                    # TODO: this won't work if a function that reads stuff is called twice, but it doesn't appear to be a common case
                    if stmt[2] in self.funcs_with_reads:
                        self.funcs_with_reads.add(get_func_name(self.func_stack[-1]))
                        self.process_invokes_with_reads.append(stmt)

                    if stmt[2] in self.funcs_with_writes:
                        self.funcs_with_writes.add(get_func_name(self.func_stack[-1]))
                        self.process_invokes_with_writes.append(stmt)

                    self.var_map = snapshot_var_map

        elif stmt[0] == '?:':
            self.push_not_impl("IO inside ternary op")
            self.crawl_stmt(stmt[2])
            self.crawl_stmt(stmt[3])
            self.crawl_stmt(stmt[4])
            self.pop_not_impl()

        elif stmt[0] == 'cast':
            ret = self.crawl_stmt(stmt[2])
            if ret is not None:
                if get_expr_type(stmt) not in (INT, REAL):
                    raise InputSchemaExtractorNotSupportedException("CAST of input to %s" % get_expr_type(stmt))
                if not ret[1].startswith('char*'):
                    return None
                #print("replacing %s / %s with %s" % (ret[1], ret[3], get_expr_type(stmt)))
                
                ret[1] = ret[1].replace('char*', get_expr_type(stmt))
                ret[3] = get_expr_type(stmt)
                return ret

        else:
            assert False, stmt[0]


    def crawl_stmt_list(self, l):
        for s in l:
            self.crawl_stmt(s)


    def crawl_func(self, func):
        self.func_stack.append(func)

        func_name = get_func_name(func)
        if func_name not in self.funcs_visited:
            self.funcs_visited[func_name] = 1
        else:
            self.funcs_visited[func_name] += 1
            if self.funcs_visited[func_name] > 10:
                self.func_stack.pop()
                return # to prevent recursion / exponential blow up
        self.crawl_stmt_list(get_func_body(func))

        self.func_stack.pop()


    def extract_schema(self, lang):
        entry_point = None
        for func_name, func in self.funcs.items():
            if lang == 'c++':
                if func_name == 'main':
                    if entry_point is not None:
                        raise InputSchemaExtractorNotSupportedException("Multiple entry points")
                    entry_point = func
            elif lang == 'java':
                if func_name.endswith(".main"):
                    args = get_func_args(func)
                    if len(args) == 1 and get_var_name(args[0]) != 'this':
                        if entry_point is not None:
                            raise InputSchemaExtractorNotSupportedException("Multiple entry points")
                        entry_point = func
            else:
                assert False

        if entry_point is None:
            raise InputSchemaExtractorNotSupportedException("Entry point not found")

        self.entry_point = entry_point

        self.push_not_impl("I/O in global initializer")
        self.crawl_func(self.funcs[GLOBALS_NAME + ".__init__"])
        self.pop_not_impl()

        self.crawl_func(entry_point)

        if not self.schema or (self.is_multi_test and len(self.schema) == 1):
            raise InputSchemaExtractorNotSupportedException("Input schema is not derived")

        if self.output_type is not None:
            if self.max_branch_out_len > 1:
                self.output_type = type_array(self.output_type)
            self.schema.append(['out', self.output_type])

        else:
            raise InputSchemaExtractorNotSupportedException("Output type is not derived")

        if self.is_multi_test:
            self.schema[0][0] = 'testN'

        # BFS to remove empty loops
        while True:
            found = False
            x = [(x, self.schema, i) for (i, x) in enumerate(self.schema)]
            for el, parent, idx in x:
                if el[0] == 'loop':
                    if not el[2]:
                        del parent[idx]
                        found = True
                        break
                    else:
                        x += [(x, el[2], i) for (i, x) in enumerate(el[2])]
            if not found:
                break

        if not self.is_multi_test:
            for k, v in self.var_map.items():
                if v[0] == 'in' and (v[1] == 'char*' or not v[1].endswith('*')) and k in self.var_map_assigns:
                    self.remove_vars.add(k)
                    self.replace_with_noops.append(self.var_map_assigns[k])
                    v[2] = k

            for k, v in self.arr_map.items():
                if v[0] == 'in' and v[1].endswith('*') and v[1] != 'char*':
                    if k in self.arr_map_inits:
                        if self.arr_map_inits[k] == False:
                            continue
                        self.replace_with_noops.append(self.arr_map_inits[k])
                    self.remove_vars.add(k)
                    self.replace_with_noops.append(self.arr_map_assigns[k])
                    self.bypassed_arrays.add(k)
                    for sz in self.schema:
                        if sz[0] == 'size':
                            for i, x in enumerate(sz[2]):
                                if x == v[2]:
                                    sz[2][i] = k
                    v[2] = k
            #print(self.arr_map)

        #print(self.arr_map, self.arr_map_assigns)

        return self.schema

    def matches_schema(self, other_schema):
        if len(self.schema) != len(other_schema):
            return False

        for our, their in zip(self.schema, other_schema):
            if our != their:
                if our[0] == 'out' and their[0] == 'out':
                    if our[1] + '*' == their[1]:
                        continue
                    if our[1] == their[1] + '*':
                        continue
                return False

        return True

    def postprocess_uast(self, desired_schema):
        assert self.matches_schema(desired_schema)

        if self.is_multi_test and (not self.multi_test_iter or not self.multi_test_var):
            raise InputSchemaExtractorNotSupportedException("Multitest schema extractor hasn't found the multitest iter or multitest var")

        for x in desired_schema:
            if x[0] == 'out':
                # it is common for schemas to be different only in whether the output is array or not
                # hence allow the caller to choose the output type
                self.output_type = x[1]

        entry_point = self.entry_point

        original_vars = (set([x[2] for x in get_func_vars(entry_point) if len(x) > 2]) | \
                        set([x[2] for x in get_func_args(entry_point) if len(x) > 2]))
        for func_name in self.funcs_with_reads:
            func = self.funcs[func_name]
            original_vars |= (set([x[2] for x in get_func_vars(func) if len(x) > 2]) | \
                              set([x[2] for x in get_func_args(func) if len(x) > 2]))
        original_vars -= set(self.remove_vars)
        def arg_name(s):
            if s == 'testN': return s
            ord_ = 0
            orig = s
            while s in original_vars:
                ord_ += 1
                s = orig + str(ord_)
            return s
        def idx_name(s):
            s = "%s_i" % arg_name(s)
            ord_ = 0
            orig = s
            while s in original_vars:
                ord_ += 1
                s = orig + str(ord_)
            return s

        for block in self.replace_with_noops:
            del block[:]
            block.append("noop")

        set_func_name(entry_point, '__main__')

        args = []
        vars_ = []
        body = []
        idx_reset = []
        body_after = []

        args_map = {}
        args_idx_map = {}
        for entry, block in self.process_input_blocks:
            del block[:]
            if entry[0] == 'size':
                arg_var = ["var", OBJECT, arg_name(entry[2][0])] # TODO: OBJECT should be the actual type
                replace_with = arg_var
                if self.is_multi_test and entry != self.multi_test_var[1]:
                    arg_var[1] += '*'
                    arg_var = ["invoke", OBJECT, 'array_index', [arg_var, self.multi_test_iter]]
                block.append("invoke")
                block.append(INT)
                block.append("len")
                block.append([arg_var])
            elif entry[0] in ['testN', 'in']:
                arg_var = ["var", entry[1], arg_name(entry[2])]
                tp = entry[1]
                replace_with = arg_var
                if self.is_multi_test and entry != self.multi_test_var[1]:
                    arg_var[1] += '*'
                    entry[1] += '*'
                    replace_with = ["invoke", tp, 'array_index', [arg_var, self.multi_test_iter]]
                    
                if entry[2] not in args_map:
                    args.append(arg_var)
                    args_map[entry[2]] = args[-1]

                if entry[2] in self.bypassed_arrays:
                    continue

                if tp.endswith("*") and tp != 'char*':
                    if entry[2] not in args_idx_map:
                        vars_.append(["var", INT, idx_name(entry[2])])
                        args_idx_map[entry[2]] = vars_[-1]
                        idx_reset.insert(0, ["assign", INT, vars_[-1], constant(INT, 0)])
                    
                    block.append("invoke")
                    block.append(tp[:-1])
                    block.append("array_index")
                    inc_idx = ["var", INT, idx_name(entry[2])]
                    inc_idx = ["assign", INT, inc_idx, ['invoke', INT, '+', [inc_idx, constant(INT, 1)]]]
                    inc_idx = ["invoke", INT, '-', [inc_idx, constant(INT, 1)]]
                    block.append([replace_with, inc_idx])

                else:
                    block[:] = replace_with


        out_type = self.output_type
        if out_type.endswith('*') and out_type != 'char*':
            vars_.append(["var", out_type, '__ret'])
            out_var = vars_[-1]

            body = [["assign", out_type, out_var, ["invoke", out_type, "_ctor", []]]] + body
            body_after += [["return", out_type, out_var]]

        for block in self.process_output_blocks:
            if block[0] == 'return': # has been processed already
                continue
            if out_type.endswith('*') and out_type != 'char*':
                block_val = block[3][0]
                del block[:]
                if out_type == 'char**':
                    if get_expr_type(block_val) == 'char*':
                        block.append('assign')
                        block.append('char**')
                        block.append(out_var)
                        if block_val[0] == "val" and '\t' not in block_val[2] and ' ' not in block_val[3]:
                            block.append(['invoke', 'char**', 'array_concat', [out_var, block_val]])
                        else:
                            block.append(['invoke', 'char**', 'array_concat', [out_var, ['invoke', 'char**', 'string_split', [block_val, ['val', 'char*', ' \\t']]]]])
                    else:
                        block.append('invoke')
                        block.append('char**')
                        block.append('array_push')
                        block.append([out_var, ['invoke', STRING, 'str', [block_val]]])
                else:
                    block.append('invoke')
                    block.append('void')
                    block.append('array_push')
                    block.append([out_var, block_val])
            else:
                assert len(block) == 4, block
                block[0] = 'return'
                if get_expr_type(block[3][0]) != 'char*' and self.output_type == 'char*':
                    block[2] = block[3][0]
                    block[2] = ['invoke', 'char*', 'str', [block[3][0]]]
                else:
                    block[2] = block[3][0]
                block.pop()

        if not self.is_multi_test:
            body = body + idx_reset
        else:
            assert self.multi_test_loop
            self.multi_test_loop[3] = idx_reset + self.multi_test_loop[3]


        misses_multi_test_iter_in_vars = self.multi_test_iter and all([x[2] != self.multi_test_iter[2] for x in get_func_vars(entry_point)])
        if misses_multi_test_iter_in_vars:
            vars_.append(self.multi_test_iter)
            body.append(["assign", INT, self.multi_test_iter, ["val", INT, 0]])

        get_func_args(entry_point)[:] = [x for x in args]
        if self.multi_test_var:
            self.multi_test_var[0][2] = arg_name(self.multi_test_var[1][2])
        get_func_vars(entry_point)[:] = [x for x in get_func_vars(entry_point) if (not self.multi_test_var or x[2] != self.multi_test_var[0][2]) and not x[2] in self.remove_vars] + vars_
        get_func_body(entry_point)[:] = body + [x for x in get_func_body(entry_point) if not (x[0] == 'while' and (len(x[3]) == 0 or (len(x[3]) == 1 and x[3][0][0] == 'noop')))] + body_after

        for func in self.funcs_with_reads:
            if get_func_name(self.funcs[func]) != '__main__':
                get_func_args(self.funcs[func])[:] = get_func_args(self.funcs[func])[:] + args + vars_
                get_func_vars(self.funcs[func])[:] = [x for x in get_func_vars(self.funcs[func]) if not x[2] in self.remove_vars]

        for func in self.funcs_with_writes:
            self.funcs[func][1] = out_type
            if self.funcs[func][0] == 'ctor':
                self.funcs[func][0] = 'func'
                self.funcs[func][2] = self.funcs[func][2].replace('.__init__', '_')
                get_func_body(self.funcs[func]).pop() # drop the return statement

            if func in self.func_returns:
                for stmt in self.func_returns[func]:
                    stmt[1] = out_type
                    stmt[2] = ["var", out_type, "__ret"]

        for invoke in self.process_invokes_with_reads:
            invoke[3] += args + vars_

        for invoke in self.process_invokes_with_writes:
            if invoke[0] == 'return': # already processed
                continue
            try:
                invoke[2] = invoke[2].replace('.__init__', '_')
            except:
                print(invoke)
                raise
            invoke[1] = out_type
            invoke[:] = ['return', VOID, [x for x in invoke]]


        return self.uast



class ExecutorContext(object):
    def __init__(self):
        super(ExecutorContext, self).__init__()
        self._registered_vars = set()
        self._vals = {}
        self._return_value = None
        self._flow_control = None

        self._instructions_count = 0


    def register_var(self, var):
        assert var[2] not in self._registered_vars, var[2]
        self._registered_vars.add(var[2])


    def set_val(self, var, val):
        assert var[2] in self._registered_vars, var
        self._vals[var[2]] = val


    def get_val(self, var):
        if var[2] not in self._vals:
            assert False, var
        return self._vals[var[2]]


def array_fill(a, b):
    for idx in range(len(a)):
        if isinstance(a, six.string_types):
            raise UASTNotImplementedException("Mutable strings")
        a[idx] = b


def map_put(a, b, c):
    a[b] = c


def map_remove_key(a, y):
    del a[y]


def array_map_clear(a):
    if isinstance(a, list):
        del a[:]
    elif isinstance(a, SortedDict):
        a.clear()
    elif isinstance(a, SortedSet):
        a.clear()
    else:
        assert False, type(a)


def array_remove_idx(a, y):
    ret = a[y]
    del a[y]
    return ret


def array_remove_value(a, y):
    y = a.index(y)
    ret = a[y]
    del a[y]
    return ret


def magic_escape(x):
    return x if x not in ['|', '\\', '+', '(', ')', ',', '[', ']'] else '\\' + x


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


DEFAULT_TYPE_FUNCS = {
    '+': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '-': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '*': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '/': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '%': lambda x, y: 'int',
    '>': lambda x, y: 'bool',
    '<': lambda x, y: 'bool',
    '>=': lambda x, y: 'bool',
    '<=': lambda x, y: 'bool',
    '==': lambda x, y: 'bool',
    '!=': lambda x, y: 'bool',
    '||': lambda x, y: 'bool',
    '&&': lambda x, y: 'bool',
    'sin': lambda x: 'real',
    'cos': lambda x: 'real',
    "str": lambda x: 'char*',
    "len": lambda x: 'int',
    "sqrt": lambda x: 'real',
    "log": lambda x: 'real',
    "ceil": lambda x: 'int',
    "sort": lambda x: x,
    "array_push": lambda x, y: 'void',
    "array_index": lambda x, y: get_array_subtype(x),
    "reverse": lambda x: x,
    "sort_cmp": lambda x, y: x,
    "concat": lambda x, y: 'char*',
    "string_find": lambda x, y: 'int',
    "string_find_last": lambda x, y: 'int',
    "string_split": lambda x, y: type_array(x),
    "map_get": lambda x, y: get_map_value_type(x),
    "map_keys": lambda x: type_array(get_map_key_type(x)),
    "map_values": lambda x: type_array(get_map_value_type(x)),
    "map_put": lambda x, y, z: 'void',
    "map_has_key": lambda x, y: 'bool',
    '!': lambda x: x,
    '~': lambda x: x,
    '&': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '|': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '^': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    '>>': lambda x, y: x,
    '<<': lambda x, y: x,
    'atan2': lambda x, y: 'real',
    'pow': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    'round': lambda x: 'int',
    'floor': lambda x: 'int',
    'clear': lambda x: 'void',
    'min': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    'max': lambda x, y: arithmetic_op_type(x, y, allow_same=True),
    'abs': lambda x: x,
    'lower': lambda x: 'char*',
    'upper': lambda x: 'char*',
    'fill': lambda x, y: 'void',
    'copy_range': lambda x, y, z: x,
    'array_index': lambda x, y: get_array_subtype(x),
    'contains': lambda x, y: 'bool',
    'string_replace_one': lambda x, y, z: x,
    'string_replace_all': lambda x, y, z: x,
    'array_concat': lambda x, y: x,
    'string_insert': lambda x, y, z: x,
    'string_trim': lambda x: x,
    'substring': lambda x, y, z: x,
    'substring_end': lambda x, y: x,
    'array_push': lambda x, y: 'void',
    'array_pop': lambda x: get_array_subtype(x),
    'array_insert': lambda x, y, z: 'void',
    'array_remove_idx': lambda x, y: get_array_subtype(x),
    'array_remove_value': lambda x, y: get_array_subtype(x),
    'array_find': lambda x, y: 'int',
    'array_find_next': lambda x, y: 'int',
    'set_push': lambda x, y: 'void',
    'set_remove': lambda x, y: 'void',
    'map_remove_key': lambda x, y: 'void',
    'array_initializer': lambda *args: type_array(args[0]),
}


# TODO: for now passing the executor for sort_cmp, might want to find a different solution later
def get_default_funcs(executor):
    funcs = {}
    funcs['=='] = lambda x, y: x == y
    funcs['!='] = lambda x, y: x != y
    funcs['&&'] = lambda x, y: x and y
    funcs['||'] = lambda x, y: x or y
    funcs['!'] = lambda x: not x
    funcs['~'] = lambda x: ~ x
    funcs['<'] = lambda x, y: x < y
    funcs['>'] = lambda x, y: x > y
    funcs['<='] = lambda x, y: x <= y
    funcs['>='] = lambda x, y: x >= y
    funcs['*'] = lambda x, y: x * y
    funcs['/'] = lambda x, y: x // y if not isinstance(x, float) and not isinstance(y, float) else x / y
    funcs['%'] = lambda x, y: x % y
    funcs['+'] = lambda x, y: x + y
    funcs['-'] = lambda x, y: x - y
    funcs['&'] = lambda x, y: x & y
    funcs['|'] = lambda x, y: x | y
    funcs['^'] = lambda x, y: x ^ y
    funcs['>>'] = lambda x, y: x >> y
    funcs['<<'] = lambda x, y: x << y
    funcs['str'] = lambda x: str(x)
    funcs['len'] = lambda x: len(x)
    funcs['sqrt'] = lambda x: math.sqrt(x)
    funcs['log'] = lambda x: math.log(x)
    funcs['atan2'] = lambda x, y: math.atan2(x, y)
    funcs['sin'] = lambda x: math.sin(x)
    funcs['cos'] = lambda x: math.cos(x)
    funcs['pow'] = lambda x, y: x ** y if y < 100 else pow(x, y, 1 << 64)
    funcs['round'] = lambda x: math.floor(x + 0.5)
    funcs['floor'] = lambda x: math.floor(x)
    funcs['ceil'] = lambda x: math.ceil(x)
    funcs['clear'] = array_map_clear
    funcs['min'] = lambda a, b: min(a, b)
    funcs['max'] = lambda a, b: max(a, b)
    funcs['abs'] = lambda a: abs(a)
    funcs['reverse'] = lambda a: list(reversed(a)) if not isinstance(a, six.string_types) else ''.join(reversed(a))
    funcs['lower'] = lambda a: a.lower()
    funcs['upper'] = lambda a: a.upper()
    funcs['sort'] = lambda a: list(sorted(a)) if not a or not isinstance(a[0], dict) else list(sorted(a, key=lambda x: tuple(x.items())))
    funcs['sort_cmp'] = lambda a, b: list(sorted(a, key=cmp_to_key(lambda x,y: executor.execute_func(b, [x,y]))))
    funcs['fill'] = array_fill
    funcs['copy_range'] = lambda arr, fr, to: [x for x in arr[fr:to]]
    funcs['array_index'] = lambda x, y: x[y] if not isinstance(x, six.string_types) else ord(x[y])
    funcs['contains'] = lambda x, y: y in x
    funcs['string_find'] = lambda x, y: x.find(y if isinstance(y, six.string_types) else chr(y))
    funcs['string_find_last'] = lambda x, y: x.rfind(y if isinstance(y, six.string_types) else chr(y))
    funcs['string_replace_one'] = lambda x, y, z: x.replace(y if isinstance(y, six.string_types) else chr(y), z if isinstance(z, six.string_types) else chr(z), 1)
    funcs['string_replace_all'] = lambda x, y, z: x.replace(y if isinstance(y, six.string_types) else chr(y), z if isinstance(z, six.string_types) else chr(z))
    funcs['concat'] = lambda x, y: (x if isinstance(x, six.string_types) else chr(x)) + (y if isinstance(y, six.string_types) else chr(y))
    funcs['array_concat'] = lambda x, y: x + y
    funcs['string_insert'] = lambda x, pos, y: x[:pos] + (y if isinstance(y, six.string_types) else chr(y)) + x[pos:]
    funcs['string_split'] = lambda x, y: [z for z in re.split('|'.join([magic_escape(_) for _ in y]), x) if z] if y != '' else [z for z in x]
    funcs['string_trim'] = lambda x: x.strip()
    funcs['substring'] = lambda x, y, z: x[y:z]
    funcs['substring_end'] = lambda x, y: x[y:]
    funcs['array_push'] = lambda x, y: x.append(y)
    funcs['array_pop'] = lambda x: x.pop()
    funcs['array_insert'] = lambda x, pos, y: x.insert(pos, y)
    funcs['array_remove_idx'] = array_remove_idx
    funcs['array_remove_value'] = array_remove_value
    funcs['array_find'] = lambda x, y: x.index(y) if y in x else -1
    funcs['array_find_next'] = lambda x, y, z: x.index(y, z) if y in x[z:] else -1
    funcs['set_push'] = lambda x, y: x.add(y)
    funcs['set_remove'] = lambda x, y: x.remove(y)
    funcs['map_has_key'] = lambda x, y: y in x
    funcs['map_put'] = map_put
    funcs['map_get'] = lambda x, y: x[y] if y in x else None
    funcs['map_keys'] = lambda x: list(x.keys())
    funcs['map_values'] = lambda x: list(x.values())
    funcs['map_remove_key'] = map_remove_key
    funcs['array_initializer'] = lambda *x: list(x)
    return funcs


class Executor(object):
    def __init__(self, data, max_total_ops=20000):
        super(Executor, self).__init__()
        self.funcs = {get_func_name(func): func for func in data['funcs']}
        self.types = {get_record_name(record): record for record in data['types']}

        self.watchers = []

        self.max_total_ops = max_total_ops
        self.total_ops = 0

        self.funcs.update(get_default_funcs(self))

        self.globals_ = {}
        global_init_func = GLOBALS_NAME + ".__init__"
        if global_init_func in self.funcs:
            self.execute_func(global_init_func, [])
        elif GLOBALS_NAME in self.types and len(self.types[GLOBALS_NAME][2]) > 2:
            raise ValueError("Must have %s if %s struct is present and non empty (%s)." % (
                global_init_func, GLOBALS_NAME, self.types[GLOBALS_NAME]
            ))

    def _observe_read(self, context, read_store, args):
        if self.watchers:
            args[-1] = (args[-1][0], tuplify(args[-1][1])) # tuplify the new value

            if read_store is not None:
                read_store[0] = args
            else:
                evt = WatcherEvent("read", self, context, args)
                self._watch(evt)

        elif read_store is not None:
            read_store[0] = []


    def _observe_write(self, context, args):
        if self.watchers:
            args[-1] = (args[-1][0], tuplify(args[-1][1])) # tuplify the new value

            evt = WatcherEvent("write", self, context, args)
            self._watch(evt)


    def register_watcher(self, watcher):
        self.watchers.append(watcher)


    def _watch(self, event):
        for watcher in self.watchers:
            watcher.watch(event)


    def compute_lhs(self, context, expr, read_store):
        assert read_store is not None and read_store[1]
        return self.compute_expression(context, expr, read_store=read_store)


    @watchable("expression")
    def compute_expression(self, context, expr, read_store=None):
        is_lhs = read_store is not None and read_store[1]

        if is_lhs and not is_assigneable(expr):
            raise UASTNotImplementedException("Non-lhs expression as argument while computing lhs")

        if expr[0] == 'assign':
            rhs = self.compute_expression(context, expr[3])
            assert is_assigneable(expr[2]), expr
            if expr[2][0] == 'var':
                # Fail if integer values are too big.
                if isinstance(rhs, int) and abs(rhs) > LARGEST_INT:
                    raise OverflowError()
                context.set_val(expr[2], rhs)
                # Same as with the field.
                inner_read_store = [None, is_lhs]
                # Calling to compute_expression to observe before_expression and after_expression events. compute
                # expression would also call to observe_read that we would prefer to skip, which we ignore here by not
                # using the contents of inner_read_store.
                self.compute_expression(context, expr[2], read_store=inner_read_store)
                self._observe_write(context, [(expr[2][2], rhs)])

            elif expr[2][0] == 'field':
                field = expr[2]
                inner_read_store = [None, True]
                record = self.compute_lhs(context, field[2], read_store=inner_read_store)
                record[field[3]] = rhs

                assert inner_read_store[0] is not None
                dependants = inner_read_store[0]
                self._observe_write(context, dependants + [(field[3], rhs)])

            elif expr[2][0] == 'invoke' and expr[2][2] == 'array_index':
                args = expr[2][3]
                deref = args[0]
                inner_read_store = [None, True]
                array = self.compute_lhs(context, args[0], read_store=inner_read_store)
                assert inner_read_store[0] is not None

                array_index = int(self.compute_expression(context, args[1]))

                assert_val_matches_type(array_index, INT)
                if isinstance(array, six.string_types):
                    # a hack way to achieve some sort of mutability in strings
                    new_val = array[:array_index] + (rhs if isinstance(rhs, six.string_types) else chr(rhs)) + array[array_index+1:]
                    self.compute_expression(context, ["assign", STRING, args[0], constant(STRING, new_val)])
                else:
                    array[array_index] = rhs

                    assert inner_read_store[0] is not None
                    dependants = inner_read_store[0]
                    self._observe_write(context, dependants + [(array_index, rhs)])

            else:
                assert False, expr
            ret = rhs
        elif expr[0] == 'var':
            ret = context.get_val(expr)
            self._observe_read(context, read_store, [(expr[2], ret)])
        elif expr[0] == 'field':
            inner_read_store = [None, is_lhs]
            obj = self.compute_expression(context, expr[2], read_store=inner_read_store)
            ret = obj[expr[3]]
            dependants = inner_read_store[0]

            if dependants is not None:
                self._observe_read(context, read_store, dependants + [(expr[3], ret)])

        elif expr[0] == 'val':
            assert_val_matches_type(expr[2], expr[1])
            ret = expr[2]
            if isinstance(ret, six.string_types):
                ret = ret.replace("\\n", "\n").replace("\\t", "\t") # TODO: proper unescaping
        elif expr[0] == 'invoke':
            if expr[2] in ['&&', '||']: # short circuiting
                larg = self.compute_expression(context, expr[3][0])
                assert type(larg) == bool
                if (larg and expr[2] == '||') or (not larg and expr[2] == '&&'):
                    ret = larg
                else:
                    ret = self.compute_expression(context, expr[3][1])

            else:
                if expr[2] == 'array_index':
                    inner_read_store = [None, is_lhs]
                    arg_vals = [self.compute_expression(context, x, read_store=inner_read_store) for x in expr[3][:1]]
                    arg_vals += [self.compute_expression(context, x) for x in expr[3][1:]]

                else:
                    try:
                        arg_vals = [self.compute_expression(context, x) for x in expr[3]]
                    except:
                        #print expr
                        raise

                if expr[2] == 'str' and expr[3][0][1] == CHAR: # TODO: fix it by replacing "str" with cast
                    ret = chr(arg_vals[0])

                elif expr[2] == '_ctor':
                    ret = self.execute_ctor(expr[1], arg_vals, expressions=expr[3])
                else:
                    try:
                        ret = self.execute_func(expr[2], arg_vals, expressions=expr[3])
                    except Exception:
                        raise

                    if expr[2] == 'array_index':
                        dependants = inner_read_store[0]
                        if dependants is not None:
                            self._observe_read(context, read_store, dependants + [(arg_vals[1], ret)])

                    if expr[2] == 'array_initializer' and expr[1] == STRING: # TODO: fix somehow
                        ret = ''.join([chr(x) for x in ret])
                    elif get_expr_type(expr) == STRING and type(ret) == list:
                        assert len(ret) == 0 or isinstance(ret[0], six.string_types), ret
                        ret = ''.join(ret)
        elif expr[0] == '?:':
            cond = self.compute_expression(context, expr[2])
            if cond:
                ret = self.compute_ternary_expression(context, expr[2], expr[3])
            else:
                ret = self.compute_ternary_expression(context, expr[2], expr[4])
        elif expr[0] == 'cast':
            assert can_cast(expr[1], expr[2][1]), expr
            ret = self.compute_expression(context, expr[2])
            if is_int_type(expr[1]):
                ret = int(float(ret))
            elif expr[1] == REAL:
                ret = float(ret)
            return ret
        else:
            raise UASTNotImplementedException("Execution of expressoin %s" % expr)
            assert False, expr
 
        try:
            assert_val_matches_type(ret, expr[1])
        except Exception as e:
            #print("Type mismatch between %s and %s while evaluating: %s (%s: %s)" % (
            #    str(ret)[:100], expr[1], expr, type(e), e), file=sys.stderr)
            #val_matches_type(ret, expr[1], True)
            raise

        if expr[1] in [REAL]:
            ret = float(ret)
        elif is_int_type(expr[1]):
            ret = int(ret)

        return ret

    
    @watchable("block")
    def execute_block(self, context, block):
        for stmt in block:
            if self.execute_statement(context, stmt):
                return True
            if context._flow_control in ['break', 'continue']:
                break
            assert context._flow_control is None
        return False

    @watchable("if_block")
    def execute_if_block(self, context, expr, block):
        # expr can be used by the watchers, e.g. for constructing the control-flow.
        return self.execute_block(context, block)

    @watchable("foreach_block")
    def execute_foreach_block(self, context, expr, block):
        # expr can be used by the watchers, e.g. for constructing the control-flow.
        return self.execute_block(context, block)

    @watchable("while_block")
    def execute_while_block(self, context, expr, block):
        # expr can be used by the watchers, e.g. for constructing the control-flow.
        return self.execute_block(context, block)

    @watchable("ternary_expression")
    def compute_ternary_expression(self, context, pred_expr, expr):
        # pred_expr can be used by the watchers, e.g. for constructing the control-flow.
        return self.compute_expression(context, expr)

    @watchable("statement")
    def execute_statement(self, context, stmt):
        context._instructions_count += 1
        self.total_ops += 1
        if self.total_ops >= self.max_total_ops:
            raise UASTTimeLimitExceeded()
        if DEBUG_INFO and context._instructions_count >= 10000 and hasattr(stmt, 'position'):
            context._instructions_count = 0
            print("DEBUG INFO: pos:", stmt.position, 'vars:', context._vals, file=sys.stderr)
            
        if stmt[0] == 'if':
            cond = self.compute_expression(context, stmt[2])
            assert isinstance(cond, bool), (cond, stmt[2])
            if cond:
                return self.execute_if_block(context, stmt[2], stmt[3])
            else:
                return self.execute_if_block(context, stmt[2], stmt[4])
        elif stmt[0] == 'foreach':
            lst = self.compute_expression(context, stmt[3])
            need_ord = isinstance(lst, six.string_types)
            for x in lst:
                context.set_val(stmt[2], x if not need_ord else ord(x))
                if self.execute_foreach_block(context, stmt[3], stmt[4]):
                    return True
                if context._flow_control == 'break':
                    context._flow_control = None
                    break
                elif context._flow_control == 'continue':
                    context._flow_control = None
        elif stmt[0] == 'while':
            while True:
                cond = self.compute_expression(context, stmt[2])
                assert isinstance(cond, bool)
                if not cond:
                    break
                if self.execute_while_block(context, stmt[2], stmt[3]):
                    return True
                if context._flow_control == 'break':
                    context._flow_control = None
                    break
                elif context._flow_control == 'continue':
                    context._flow_control = None

                assert not self.execute_while_block(context, stmt[2], stmt[4])
        elif stmt[0] == 'break':
            context._flow_control = 'break'
            return False
        elif stmt[0] == 'continue':
            context._flow_control = 'continue'
            return False
        elif stmt[0] == 'return':
            context._return_value = self.compute_expression(context, stmt[2])
            return True
        elif stmt[0] == 'noop':
            return False
        else:
            self.compute_expression(context, stmt)


    def execute_ctor(self, ret_type, args, expressions):
        if ret_type.endswith("*"):
            if len(args) == 0:
                return [] if ret_type != 'char*' else ""
            elif len(args) == 1 and not val_matches_type(args[0], INT):
                # initialize with the first argument
                return list(args[0])
            else:
                assert len(ret_type) > len(args) and all([x == '*' for x in ret_type[-len(args):]]), "TYPE: %s, ARGS: %s" % (ret_type, args)
                subtype = ret_type
                for arg in args:
                    assert_val_matches_type(arg, INT)
                    subtype = get_array_subtype(subtype)

                # We measured the size of the N-dimensional array initialized with default values of different types and
                # measured the approx number of bytes used by each element. Based on this we cut the maximum array size
                # that we can initialize.
                approx_memory_overhead = {
                    INT: 8,
                    REAL: 8,
                    CHAR: 4,
                    STRING: 4,
                    BOOL: 1
                }
                memory_cutoff = 10*2**20  # Allocate no more than 10MiB during array initialization.
                assert functools.reduce(mul, args) * approx_memory_overhead[subtype] <= memory_cutoff, (
                    "CTOR allocates too much memory %s %s, %s" % (ret_type, args, expressions))
                return np.full(tuple(args), default_value(subtype)).tolist()
        elif ret_type.endswith("%"):
            return SortedSet() if len(args) == 0 else SortedSet(args[0])
        elif ret_type.endswith('>'):
            return SortedDict()
        elif ret_type == INT:
            assert len(args) == 0
            return 0
        elif ret_type.endswith('#'):
            return self.execute_func(ret_type[:-1] + ".__init__", args, expressions=expressions)
        else:
            assert False, ret_type

    @watchable('func_block')
    def execute_func_block(self, context, func_name, func_vars, func_args, args_vals, expressions, block):
        # func_name, func_vars, func_args, args_vals and expressions can be used by the watchers, e.g. for constructing
        # the data-flow.
        assert len(func_args) == len(args_vals)
        assert expressions is None or len(expressions) == len(func_args)
        self.execute_block(context, block)

    def execute_func(self, func_name, args, tolerate_missing_this=False, expressions=None):
        context = ExecutorContext()

        if func_name not in self.funcs:
            raise UASTNotImplementedException("Interpreter function %s" % func_name)

        func = self.funcs[func_name]

        if callable(func):
            try:
                return func(*args)
            except Exception:
                # print(func_name, args)
                raise

        if self.watchers:
            self._watch(WatcherEvent("before_func", self, context, func, args))

        globals_var = var(GLOBALS_NAME, func[1])
        context.register_var(globals_var)
        context.set_val(globals_var, self.globals_)

        if func[0] == 'ctor':
            ctor_type_name = type_to_record_name(func[1])
            ctor_type = self.types[ctor_type_name]
            ret_var = var("this", func[1])
            context.register_var(ret_var)
            context.set_val(ret_var, {})

        if tolerate_missing_this and len(args) == len(get_func_args(func)) + 1:
            args = args[1:]

        if len(args) != len(get_func_args(func)):
            #print >> sys.stderr, func
            #print >> sys.stderr, args
            #print >> sys.stderr, get_func_args(func)
            raise UASTNotImplementedException("Polymorphism (len(%s) <> len(%s) when calling %s)" % (args, get_func_args(func), get_func_name(func)))

        for arg, arg_def in zip(args, get_func_args(func)):
            assert_val_matches_type(arg, get_expr_type(arg_def))
            context.register_var(arg_def)
            context.set_val(arg_def, arg)

        for var_ in get_func_vars(func):
            context.register_var(var_)

        self.execute_func_block(context, func_name, get_func_vars(func), get_func_args(func), args, expressions, get_func_body(func))

        assert_val_matches_type(context._return_value, get_func_return_type(func))

        if self.watchers:
            self._watch(WatcherEvent("after_func", self, context, context._return_value, func, args))

        return context._return_value
