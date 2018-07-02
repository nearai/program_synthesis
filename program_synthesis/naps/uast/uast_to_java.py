import collections


INDENT = "  "
BINARY_FUNCS = ['+', '-', '*', '/', '%', '&',
                '&&', '||', '|', '<', '>', '>=', '<=', '==']


def _convert_global_name(name):
    return name[name.rfind('.') + 1:]


def _convert_var_name(name):
    if name == "this":
        return "this_"
    return name.replace('.', '_')


def _convert_expression(expr):
    if expr[0] == "assign":
        return "%s = %s" % (_convert_expression(expr[2]), _convert_expression(expr[3]))
    if expr[0] == "var":
        return _convert_var_name(expr[2])
    if expr[0] == "field":
        obj = _convert_expression(expr[2])
        if obj == "__globals__":
            return _convert_global_name(expr[3])
        return obj + "." + _convert_var_name(expr[3])
    if expr[0] == "val":
        if isinstance(expr[2], bool):
            return 'TRUE' if expr[2] else "FALSE"
        if isinstance(expr[2], str):
            return '"%s"' % expr[2]
        return str(expr[2])
    if expr[0] == "cast":
        return "((%s)%s)" % (expr[1], _convert_expression(expr[2]))
    if expr[0] == "?:":
        return "(%s ? %s : %s)" % tuple([_convert_expression(e) for e in expr[2:]])
    if expr[0] == "invoke":
        args = [_convert_expression(arg) for arg in expr[3]]
        func = expr[2]
        if func in BINARY_FUNCS and len(args) == 2:
            return "%s %s %s" % (args[0], func, args[1])
        if func == "array_index" and len(args) == 2:
            if expr[3][0][1] == "char*":
                return "%s.charAt(%s)" % (args[0], args[1])
            return "%s[%s]" % (args[0], args[1])
        if func == "_ctor":
            return "new %s(%s)" % (_convert_type(expr[1]), ', '.join(args))
        if func.endswith('.__init__'):
            return "new %s(%s)" % (func[:-9], ', '.join(args))
        if func == "array_initializer":
            return "new %s {%s}" % (_convert_type(expr[1]), ', '.join(args))
        if func == "len":
            return "%s.length()" % args[0]
        return "%s(%s)" % (func, ', '.join(args))
    raise ValueError("Unknown expression kind: %s (%s)" % (expr[0], expr[1:]))


def _convert_statement(statement, indent=0):
    if statement[0] == "if":
        cond = _convert_expression(statement[2])
        then_ = _convert_block(statement[3], indent)
        else_ = _convert_block(statement[4], indent)
        if else_:
            return INDENT * indent + "if (%s) %s%selse %s" % (cond, then_, INDENT * indent, else_)
        return INDENT * indent + "if (%s) %s" % (cond, then_)
    if statement[0] == "while":
        cond = _convert_expression(statement[2])
        body = _convert_block(statement[3], indent, False)
        increment = _convert_block(statement[4], indent, False)
        return INDENT * indent + "while (%s) {\n%s%s%s}\n" % (cond, body, increment, INDENT * indent)
    if statement[0] == "return":
        return INDENT * indent + "return %s;\n" % _convert_expression(statement[2])
    if statement[0] == "foreach":
        var1 = _convert_expression(statement[2])
        var2 = _convert_expression(statement[3])
        body_ = _convert_block(statement[4], 0, False)
        return INDENT * indent + "for (%s %s : %s) %s" % (_convert_type(statement[2][1]), var1, var2, body_)
    if statement[0] == "noop":
        return ""
    if statement[0] == "break":
        return INDENT * indent + "break;"
    if statement[0] == "continue":
        return INDENT * indent + "continue;"
    # Single statement of variable is not useful.
    if statement[0] == "var":
        return ""
    return INDENT * indent + _convert_expression(statement) + ";\n"


def _convert_block(body, indent=0, bracket=True, variables=None):
    result = []
    if variables:
        for _, type_, name in variables:
            result.append("  %s %s;\n" %
                        (_convert_type(type_), _convert_var_name(name)))
    for statement in body:
        result.append(_convert_statement(statement, indent=indent + 1))
    result = "".join(result)
    if bracket:
        result = "{\n%s%s}\n" % (
            result, INDENT * indent
        )
    return result


def _convert_type(type_):
    if type_[-1] == '#':
        return _convert_type(type_[:-1])
    if type_ == "char*":
        return "String"
    if type_[-1] == "*":
        return _convert_type(type_[:-1]) + "[]"
    return type_


def _convert_args(arguments):
    result = []
    for _, type_, name in arguments:
        result.append("%s %s" % (_convert_type(type_), _convert_var_name(name)))
    return ', '.join(result)


def _convert_func(kind, return_type, name, arguments, variables, body):
    args = _convert_args(arguments)
    if kind == 'ctor':
        if len(body) == 1:
            return ""
        return "public %s(%s) %s" % (_convert_type(return_type), args, _convert_block(body[:-1], variables=variables))
    return "public static %s %s(%s) %s" % (
        _convert_type(return_type), name, args, _convert_block(body, variables=variables)
    )


def uast_to_java(uast):
    libraries = ["import java.util.*;"]
    static_inits = collections.defaultdict(dict)
    for kind, _, name, _, _, body in uast['funcs']:
        if name == '__globals__.__init__':
            for statement in body:
                assert statement[0] == "assign", "Unrecognized format of static inits"
                cls_name, name = statement[2][3].split('.')
                static_inits[cls_name][name] = _convert_expression(statement[3])
    classes = collections.defaultdict(lambda: collections.defaultdict(list))
    default_cls = '__globals__'
    for _, name, fields in uast['types']:
        if name == "__globals__":
            for fullname, (_, type_, name) in fields.items():
                cls_name = fullname[:fullname.rfind('.')]
                value = " = %s" % static_inits[cls_name][name] if static_inits[cls_name][name] else ""
                classes[cls_name]['fields'].append("static %s %s%s;" % (_convert_type(type_), name, value))
        else:
            classes[name]['fields'] += ["%s %s;" % (type_, _convert_var_name(var_name)) for var_name, (_, type_, var) in fields.items()]
            if default_cls == "__globals__":
                default_cls = name
    for kind, return_type, name, arguments, variables, body in uast['funcs']:
        if name == "__globals__.__init__":
            continue
        into_cls = default_cls
        for cls_name in classes.keys():
            if name.startswith(cls_name + "."):
                into_cls = cls_name
                break
        if '.' in name:
            name = name[name.find('.') + 1:]
        classes[into_cls]['funcs'].append(_convert_func(kind, return_type, name, arguments, variables, body))
    result = "\n".join(libraries)
    for name, class_ in classes.items():
        result += "\nclass %s {\n%s\n%s}\n" % (name, '\n'.join(class_['fields']), '\n'.join(class_['funcs']))
    return result
