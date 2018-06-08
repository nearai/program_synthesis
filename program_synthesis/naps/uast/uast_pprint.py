import six


INDENT = "  "


def pformat_expression(s):
    def pformat_invoke(type_, func, args):
        args = [pformat_expression(arg) for arg in args]
        if func in ('+', '-', '*', '/', '%', '!=', '==', '<=', '>=', '>', '<', '&&', '||', '&', '|', '^', '<<', '>>'):
            return "(%s %s %s)" % (args[0], func, args[1])
        if func == 'array_index':
            return "%s[%s]" % (args[0], args[1])
        if func == 'array_initializer':
            return "new %s[]{%s}" % (type_, ', '.join(args))
        if func == '_ctor':
            return "new %s(%s)" % (type_, ', '.join(args))
        if func.endswith('.__init__'):
            return "new %s(%s)" % (func[:-9], ', '.join(args))

        return "%s(%s)" % (func, ', '.join(args))
    _TYPES = {
        "assign": lambda _, left, right: "%s = %s" % (pformat_expression(left), pformat_expression(right)),
        "val": lambda _, x: '"%s"' % x if isinstance(x, six.string_types) else str(x),
        "var": lambda _, x: x,
        "field": lambda _, expr, name: pformat_expression(expr) + "." + name,
        "cast": lambda type_, x: "(%s)%s" % (type_, pformat_expression(x)),
        "invoke": pformat_invoke,
        "?:": lambda _, cond, left, right: "(%s)?(%s):(%s)" % (pformat_expression(cond), pformat_expression(left), pformat_expression(right))
    }
    return _TYPES[s[0]](*s[1:])


def pformat_statement(s, indent=0):
    _TYPES = {
        "while": lambda cond, body, inc: "for(; %s; %s)\n%s\n" % (
            pformat_expression(cond),
            ','.join([pformat_expression(x) for x in inc]),
            pformat_block(body, indent + 1)),
        "foreach": lambda var1, var2, body: "foreach(%s : %s)\n%s" % (
            pformat_expression(var1), pformat_expression(var2), pformat_block(body, indent + 1)),
        "return": lambda expr: "return %s\n" % pformat_expression(expr),
        "if": lambda cond, then_, else_: "if %s\n%s%selse\n%s" % (
            pformat_expression(cond),
            pformat_block(then_, indent + 1),
            INDENT * indent,
            pformat_block(else_, indent + 1)
        ),
        "noop": lambda: "noop\n",
        "break": lambda: "break\n",
        "continue": lambda: "continue\n",
    }
    if s[0] not in _TYPES:
        return INDENT * indent + pformat_expression(s) + "\n"
    return INDENT * indent + _TYPES[s[0]](*s[2:])

def pformat_block(block, indent=0):
    res = ""
    for s in block:
        res += pformat_statement(s, indent=indent)
    if not block:
        res += INDENT * indent + "pass\n"
    return res


def pformat_struct(struct):
    fields = ["%s %s" % (args[1], name) for name, args in struct[2].items()]
    return "%s %s (%s)\n" % (struct[0], struct[1], ', '.join(fields))


def pformat_func(func):
    kind, return_type, func_name, args, variables, body = func
    args_ = ', '.join(['%s %s' % (type_, name) for _, type_, name in args])
    res = '%s %s %s(%s)\n' % (kind, return_type, func_name, args_)
    if variables:
        res += INDENT + \
            'vars: %s\n' % ', '.join(
                [' % s % s' % (type_, name) for _, type_, name in variables])
    return res + pformat_block(body, 1)


def pformat(code_tree):
    res = ""
    for struct in code_tree['types']:
        res += pformat_struct(struct)
    for func in code_tree['funcs']:
        res += pformat_func(func)
    return res


def pprint(code_tree):
    print(pformat(code_tree))
