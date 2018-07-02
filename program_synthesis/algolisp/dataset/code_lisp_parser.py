import re

from program_synthesis.algolisp.dataset import code_types
from program_synthesis.algolisp.dataset import code_lisp


def parse_arguments(args):
    res = []
    for arg in args.split(","):
        type_, name = arg.strip().split(" ")
        res.append((name.strip(), code_types.str_to_type(type_.strip())))
    return res


def parse_statement(program):
    def _parse_args(args):
        cnt = 0
        res, cur = [], ""
        for i in range(len(args)):
            if args[i] == "(":
                cnt += 1
            elif args[i] == ")":
                cnt -= 1
            elif args[i] == ",":
                if cnt == 0:
                    if cur:
                        res.append(cur.strip())
                        cur = ""
                    continue
            cur += args[i]
        if cur:
            res.append(cur.strip())
        return res
            
    res = re.match("([^\(\)]*)\((.*)\)", program)
    if res:
        func, args = res.groups()
        args = _parse_args(args)
        lst = [parse_statement(arg) for arg in args]
        return [func] + lst
    return program


def parse(code):
    lines = code.split("\n")
    for line in lines:
        res = re.match("def ([a-zA-Z0-9_]+)\(([^\)]*)\)\:([^\s]*)\s(.*)", line)
        if res:
            name, args, return_type, program = res.groups()
            args = parse_arguments(args)
            return_type = code_types.str_to_type(return_type)
            program = parse_statement(program)
            yield name, program, args, return_type


def parse_and_compile(lisp_units, code):
    funcs = {}
    for name, program, args, return_type in parse(code):
        funcs[name] = code_lisp.compile_func(lisp_units, name, program, args, return_type)
    return funcs
