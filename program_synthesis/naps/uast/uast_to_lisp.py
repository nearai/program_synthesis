import codecs
import six
import json

from . import uast


_DEFAULT_NAMES = {
    'var': ('this', '__globals__'),
    'func': list(uast.DEFAULT_TYPE_FUNCS.keys()) + [
             '__main__', '__globals__.__init__', '_ctor'],
    'struct': ('void', 'int', 'char', 'real', '__globals__', 'bool', 'object')
}

STRING_SEPARATOR = '|'


def rearrange_funcs(tree):
    # Rearrange functions such that __main__ goes last.
    funcs, main_func = [], None
    for func in tree['funcs']:
        if func[2] != '__main__':
            funcs.append(func)
        else:
            main_func = func
    assert main_func is not None
    funcs.append(main_func)
    return funcs


def remap_uast(code_tree, names=None):
    def remap_vars(args):
        res = []
        for _, type_, name in args:
            if name not in _DEFAULT_NAMES['var']:
                names['var'].setdefault(name, 'var%d' % len(names['var']))
            res.append(['var', remap_type(type_), names['var'].get(name, name)])
        return res

    def remap_type(tp):
        if tp[-1] == '#':
            return remap_type(tp[:-1]) + '#'
        if tp[-1] in ('*', '%'):
            return remap_type(tp[:-1]) + tp[-1]
        if tp[0] == '<' and tp[-1] == '>':
            return '<%s|%s>' % (remap_type(uast.get_map_key_type(tp)), remap_type(uast.get_map_value_type(tp)))
        return names['struct'].get(tp, tp)

    def remap_expression(expr, sort_cmp_flag=False):
        TYPES = {
            'assign': lambda type_, left, right: ['assign', remap_type(type_), remap_expression(left), remap_expression(right)],
            'var': lambda type_, name: ['var', remap_type(type_), names['var'].get(name, name)],
            'field': lambda type_, obj, name: ['field', remap_type(type_), remap_expression(obj), names['var'].get(name, name)],
            'val': lambda type_, val: ['val', remap_type(type_), val if not sort_cmp_flag else names['func'].get(val, val)],
            'cast': lambda type_, e: ['cast', remap_type(type_), remap_expression(e)],
            'invoke': lambda type_, name, args: ['invoke', remap_type(type_), names['func'].get(name, name), [remap_expression(arg, sort_cmp_flag=(name=='sort_cmp')) for arg in args]],
            '?:': lambda type_, cond, then_, else_: ['?:', remap_type(type_), remap_expression(cond), remap_expression(then_), remap_expression(else_)],
        }
        if expr[0] in TYPES:
            return TYPES[expr[0]](*expr[1:])
        raise ValueError("WTF? ", expr)

    def remap_statement(statement):
        TYPES = {
            'while': lambda _, cond, body, increment: ['while', 'void', remap_expression(cond), remap_block(body), remap_block(increment)],
            'foreach': lambda _, var1, var2, body: ['foreach', 'void', remap_expression(var1), remap_expression(var2), remap_block(body)],
            'if': lambda _, cond, then_, else_: ['if', 'void', remap_expression(cond), remap_block(then_), remap_block(else_)],
            'return': lambda _, expr: ['return', 'void', remap_expression(expr)],
            'break': lambda _: ['break', 'void'],
            'continue': lambda _: ['continue', 'void'],
            'noop': lambda: ['noop'],
        }
        if statement[0] in TYPES:
            return TYPES[statement[0]](*statement[1:])
        return remap_expression(statement)

    def remap_block(block):
        res = []
        for statement in block:
            res.append(remap_statement(statement))
        return res

    if names is None:
        names = {'struct': {}, 'func': {}, 'var': {}}

    for kind, struct_name, fields in code_tree['types']:
        if struct_name not in _DEFAULT_NAMES['struct']:
            names['struct'][struct_name] = 'struct%d' % len(names['struct'])
    structs = []
    for kind, struct_name, fields in code_tree['types']:
        new_fields = {}
        for name, (_, type_, _) in fields.items():
            names['var'].setdefault(name, 'var%d' % len(names['var']))
            new_fields[names['var'][name]] = ['var', remap_type(type_), names['var'][name]]
        structs.append([kind, names['struct'].get(struct_name, struct_name), new_fields])
    orig_funcs = rearrange_funcs(code_tree)
    for func in orig_funcs:
        if func[2] in ('__main__', '__globals__.__init__'):
            continue
        if func[0] == 'func':
            names['func'][func[2]] = 'func%d' % len(names['func'])
        else:
            rt = remap_type(func[1])
            names['func'][func[2]] = '%s.__init__' % rt[:-1]
    funcs = []
    for func in orig_funcs:
        args = remap_vars(func[3])
        variables = remap_vars(func[4])
        body = remap_block(func[5])
        funcs.append([func[0], remap_type(func[1]), names['func'].get(
            func[2], func[2]), args, variables, body])
    return {"types": structs, "funcs": funcs}


def unescape_string(s):
    # return s.replace('||', ' ').replace('\\|', '|')
    return codecs.escape_decode(s.replace('||', ' ').replace('\\|', '|'))[0].decode("utf-8")


def escape_string(s):
    # Note, json.dumps('\n') == '"\\n"'
    return json.dumps(s.replace('|', '\\|').replace(' ', '||'))


def _convert_names(name, kind, names):
    if name in _DEFAULT_NAMES[kind]:
        if name == '__globals__.__init__':
            name = '__globals_____init__'
        return name
    if name.endswith('#'):
        return _convert_names(name[:-1], kind, names)
    if name.endswith('*') or name.endswith('%'):
        return _convert_names(name[:-1], kind, names) + name[-1]
    if name.startswith('<') and name.endswith('>') and '|' in name:
        key_, value_ = uast.get_map_key_type(name), uast.get_map_value_type(name)
        return '<%s|%s>' % (_convert_names(key_, kind, names), _convert_names(value_, kind, names))
    if name not in names[kind]:
        names[kind][name] = '%s%d' % (kind, len(names[kind]))
    return names[kind][name]


def _convert_expression(expression, names, is_sort_cmp=False):
    if expression[0] == 'assign':
        return '(= %s %s)' % (_convert_expression(expression[2], names), _convert_expression(expression[3], names))
    if expression[0] == 'var':
        return _convert_names(expression[2], 'var', names)
    if expression[0] == 'field':
        return '(field %s %s)' % (_convert_expression(expression[2], names), _convert_names(expression[3], 'var', names))
    if expression[0] == 'val':
        if isinstance(expression[2], six.string_types):
            if is_sort_cmp:
                return _convert_names(expression[2], 'func', names)
            return escape_string(expression[2])
        if expression[1] == 'real':
            val = str(float(expression[2]))
            return val.replace('-', '_').replace('+', '__')
        return str(expression[2])
    if expression[0] == 'cast':
        return '(cast %s %s)' % (
            _convert_names(expression[1], 'struct', names),
            _convert_expression(expression[2], names))
    if expression[0] == 'invoke':
        args = [_convert_names(expression[2], 'func', names)] + \
            [_convert_expression(arg, names, expression[2] == "sort_cmp") for arg in expression[3]]
        if expression[2] == '_ctor':
            args.insert(1, _convert_names(expression[1], 'struct', names))
        return '(%s)' % ' '.join(args)
    if expression[0] == '?:':
        return '(? %s %s %s)' % tuple([
            _convert_expression(exp, names) for exp in expression[2:]
        ])
    if expression[0] == 'noop':
        return '(noop)'
    raise ValueError("Unknown expression kind: %s (%s)" %
                     (expression[0], expression))


def _convert_statement(statement, names):
    if statement[0] == 'if':
        cond = _convert_expression(statement[2], names)
        then_ = _convert_block(statement[3], names)
        else_ = _convert_block(statement[4], names)
        return '(if %s %s %s)' % (cond, then_, else_)
    if statement[0] == 'while':
        cond = _convert_expression(statement[2], names)
        body_ = _convert_block(statement[3], names)
        increment = _convert_block(statement[4], names)
        return '(while %s %s %s)' % (cond, body_, increment)
    if statement[0] == 'foreach':
        var1 = _convert_expression(statement[2], names)
        var2 = _convert_expression(statement[3], names)
        body_ = _convert_block(statement[4], names)
        return '(foreach %s %s %s)' % (var1, var2, body_)
    if statement[0] == 'return':
        return '(return %s)' % (_convert_expression(statement[2], names))
    if statement[0] in ('break', 'continue'):
        return '(%s)' % (statement[0])
    return _convert_expression(statement, names)


def _convert_block(statements, names):
    result = []
    for statement in statements:
        result.append(_convert_statement(statement, names))
    return '[%s]' % ' '.join(result)


def _uast_to_lisp(code_tree, names):
    result = "("
    for _, struct_name, fields in code_tree['types']:
        _convert_names(struct_name, 'struct', names)
    for _, struct_name, fields in code_tree['types']:
        fields_ = []
        for name, field in sorted(fields.items()):
            fields_.append("(%s %s)" %
                           (_convert_names(name, 'var', names), _convert_names(field[1], 'struct', names)))
        result += "(struct %s [%s]) " % (_convert_names(struct_name,
                                                        'struct', names), ' '.join(fields_))

    orig_funcs = rearrange_funcs(code_tree)
    for kind, return_type, func_name, args, variables, body in orig_funcs:
        args_ = []
        for _, type_, name in args:
            args_.append("(%s %s)" % (
                _convert_names(name, 'var', names),
                _convert_names(type_, 'struct', names)))
        variables_ = []
        for _, type_, name in variables:
            variables_.append("(%s %s)" % (
                _convert_names(name, 'var', names),
                _convert_names(type_, 'struct', names)))
        body_ = _convert_block(body, names)
        result += "(%s %s %s [%s] [%s] %s) " % (
            kind, _convert_names(func_name, 'func', names),
            _convert_names(return_type, 'struct', names), ' '.join(args_),
            ' '.join(variables_), body_
        )
    result += ")"
    return result


def uast_to_lisp_body(code_tree, anonymize=True, names=None):
    assert anonymize
    if names is None:
        names = {'struct': {}, 'func': {}, 'var': {}}
    return _convert_block(code_tree, names)


def uast_to_lisp(code_tree, anonymize=True):
    class identity(object):
        def get(self, key, default):
            return default
        def __getitem__(self, key):
            return key
        def setdefault(self, key, default):
            pass
        def __contains__(self, value):
            return True
    if anonymize:
        names = {'struct': {}, 'func': {}, 'var': {}}
    else:
        names = {'struct': identity(), 'func': identity(), 'var': identity()}
    return _uast_to_lisp(code_tree, names)


def flatten_tree(tree):
    EOS = ["<\S>"]

    def _flatten(lst_of_lsts):
        res = []
        for lst in lst_of_lsts:
            res.extend(lst)
        return res

    def _flatten_var(var):
        return [var[1], var[2]]

    def _flatten_var_list(var_list):
        return _flatten([_flatten_var(var) for var in var_list]) + EOS

    def list_concat(lst):
        res = []
        for x in lst:
            res.extend(x)
        return res

    def _flatten_expression(expression):
        TYPES_ = {
            'assign': lambda _, left, right: ["assign"] + _flatten_expression(left) + _flatten_expression(right),
            'var': lambda _, name: [name],
            'field': lambda _, obj, name: ["."] + _flatten_expression(obj) + [name],
            'val': lambda _, val: [val],
            'cast': lambda type_, expr: ["cast", type_] + _flatten_expression(expr),
            'invoke': lambda _, name, args: [name] + list_concat([_flatten_expression(arg) for arg in args]),
            '?:': lambda _, *args: ["?:"] + list_concat([_flatten_expression(arg) for arg in args]),
        }
        return TYPES_[expression[0]](*expression[1:])

    def _flatten_statement(statement):
        TYPES_ = {
            'if': lambda cond, then_, else_: ["if"] + _flatten_expression(cond) + _flatten_body(then_) + _flatten_body(else_),
            'while': lambda cond, body, increment: ["while"] + _flatten_expression(cond) + _flatten_body(body) + _flatten_body(increment),
            'foreach': lambda var1, var2, body: ["foreach"] + _flatten_expression(var1) + _flatten_expression(var2) + _flatten_body(body),
            'return': lambda expr: ["return"] + _flatten_expression(expr)
        }
        if statement[0] not in TYPES_:
            return _flatten_expression(statement)
        return TYPES_[statement[0]](*statement[2:])

    def _flatten_body(body):
        res = []
        for statement in body:
            res.extend(_flatten_statement(statement))
        return res + EOS

    orig_funcs = rearrange_funcs(tree)
    result = []
    for kind, return_type, func_name, args, variables, body in orig_funcs:
        args = _flatten_var_list(args)
        variables = _flatten_var_list(variables)
        body = _flatten_body(body)
        result.extend([kind, return_type, func_name] + args + variables + body)
    return result
