

def walk_uast(tree, on_tree=None, on_struct=None, on_func=None, on_block=None, on_statement=None, on_expression=None):
    def _walk_expression(expression, is_lhs=False):
        TYPES_ = {
            'assign': lambda left, right: (_walk_expression(left, True), _walk_expression(right)),
            'var': lambda name: (name,),
            'field': lambda obj, name: (_walk_expression(obj, is_lhs),),
            'val': lambda val: (val,),
            'cast': lambda expr: _walk_expression(expr),
            'invoke': lambda name, args: [name] + [_walk_expression(arg, is_lhs and arg_ord == 0) for (arg_ord, arg) in enumerate(args)],
            '?:': lambda *args: [_walk_expression(arg) for arg in args],
        }
        if expression[0] not in TYPES_:
            raise ValueError("Unknown expression kind: %s (%s)" % (expression[0], expression))
        res = TYPES_[expression[0]](*expression[2:])
        return on_expression(expression, is_lhs, res) if on_expression else False

    def _walk_statement(statement):
        TYPES_ = {
            'if': lambda cond, then_, else_: (_walk_expression(cond), _walk_block(then_), _walk_block(else_)),
            'while': lambda cond, body, increment: (_walk_expression(cond), _walk_block(body), _walk_block(increment)),
            'foreach': lambda var1, var2, body: (_walk_expression(var1), _walk_expression(var2), _walk_block(body)),
            'return': lambda expr: _walk_expression(expr),
            'break': lambda: 'break',
            'continue': lambda: 'continue',
            'noop': lambda: 'noop',
        }
        if statement[0] not in TYPES_:
            res = _walk_expression(statement)
        else:
            res = TYPES_[statement[0]](*statement[2:])
        return on_statement(statement, res) if on_statement else False

    def _walk_block(block):
        res = []
        for statement in block:
            res.append(_walk_statement(statement))
        return on_block(block, res) if on_block else False

    structs, funcs = [], []
    for kind, name, fields in tree['types']:
        structs.append(on_struct and on_struct(name, fields))
    for kind, return_type, func_name, args, variables, body in tree['funcs']:
        res = _walk_block(body)
        funcs.append(on_func(kind, return_type, func_name, args, variables, body, res) if on_func else False)
    return on_tree(tree, structs, funcs) if on_tree else False


# if __name__ == "__main__":
#     t = {"funcs": [["func", "void", "__globals__.__init__", [], [], [["assign", "int", ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"], ["invoke", "int", "+", [["invoke", "int", "*", [["invoke", "int", "*", [["val", "int", 1000], ["val", "int", 1000]]], ["val", "int", 1000]]], ["val", "int", 7]]]]]], ["ctor", "ParenthesisRemoval#", "ParenthesisRemoval.__init__", [], [], [["return", "void", ["var", "ParenthesisRemoval#", "this"]]]], ["func", "int", "ParenthesisRemoval.countWays", [["var", "ParenthesisRemoval#", "this"], ["var", "char*", "s"]], [["var", "int", "res"], ["var", "int", "cnt"], ["var", "int", "i"]], [["assign", "int", ["var", "int", "res"], ["val", "int", 1]], ["assign", "int", ["var", "int", "cnt"], ["val", "int", 0]], ["assign", "int", ["var", "int", "i"], ["val", "int", 0]], ["while", "void", ["invoke", "bool", "<", [["var", "int", "i"], ["invoke", "int", "len", [["var", "char*", "s"]]]]], [["if", "void", ["invoke", "bool", "==", [["invoke", "char", "array_index", [["var", "char*", "s"], ["var", "int", "i"]]], ["val", "char", 40]]], [["assign", "int", ["var", "int", "cnt"], ["invoke", "int", "+", [["var", "int", "cnt"], ["val", "int", 1]]]], ["var", "int", "cnt"]], [["assign", "int", ["var", "int", "res"], ["invoke", "int", "*", [["var", "int", "res"], ["cast", "int", ["var", "int", "cnt"]]]]], ["assign", "int", ["var", "int", "res"], ["invoke", "int", "%", [["var", "int", "res"], ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"]]]], ["assign", "int", ["var", "int", "cnt"], ["invoke", "int", "-", [["var", "int", "cnt"], ["val", "int", 1]]]], ["var", "int", "cnt"]]]], [["assign", "int", ["var", "int", "i"], ["invoke", "int", "+", [["var", "int", "i"], ["val", "int", 1]]]], ["var", "int", "i"]]], ["return", "void", ["cast", "int", ["invoke", "int", "%", [["var", "int", "res"], ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"]]]]]]], ["func", "int", "__main__", [["var", "char*", "s"]], [], [["return", "void", ["invoke", "int", "ParenthesisRemoval.countWays", [["invoke", "ParenthesisRemoval#", "ParenthesisRemoval.__init__", []], ["var", "char*", "s"]]]]]]], "types": [["record", "__globals__", {"ParenthesisRemoval.mod": ["var", "int", "mod"]}], ["record", "ParenthesisRemoval", {}]]}
#     result = []
#     walk_uast(t, on_statement=lambda s, *args: result.append(s))
#     for expression in result:
#         print(expression)
