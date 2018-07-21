from program_synthesis.naps.uast.uast_to_cpp.type_converter import to_cpp_type
from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import CPPLibs
from program_synthesis.naps.uast import uast


def to_cpp_expr(expr, libs):
    car = expr[0]
    if car == 'assign':
        return "%s = (%s)" % (to_cpp_expr(expr[2], libs), to_cpp_expr(expr[3], libs))
    elif car == 'var':
        return expr[2]
    elif car == 'field':
        inst = to_cpp_expr(expr[2], libs)
        return "(%s)->%s" % (inst, expr[3])
    elif car == 'val':
        if expr[1] == 'real':
            return 'static_cast<double>(%s)' % expr[2]
        elif expr[1] == 'char*':
            return '"%s"' % expr[2]
        return expr[2]
    elif car == '?:':
        expr1 = to_cpp_expr(expr[2], libs)
        expr2 = to_cpp_expr(expr[3], libs)
        expr3 = to_cpp_expr(expr[4], libs)
        return "(%s)?(%s):(%s)" % (expr1, expr2, expr3)
    elif car == 'cast':
        t = to_cpp_type(expr[1], libs)
        e = to_cpp_expr(expr[2], libs)
        return "static_cast< %s > (%s)" % (t, e)
    elif car == 'invoke':
        return to_cpp_invoke(expr, libs)


def to_cpp_args(libs, *args):
    res = []
    for arg in args:
        res.append(to_cpp_expr(arg, libs))
    return tuple(res)


def to_cpp_invoke(expr, libs):
    op = expr[2]
    args = expr[3]
    if op in ('==', '!=', '&&', '||', '<', '>', '<=', '>=', '*',
              '%', '+', '-', '&', '|', '^', '>>', '<<', '/'):
        return "(%s) %s (%s)" % (to_cpp_expr(args[0], libs), op, to_cpp_expr(args[1], libs))
    elif op in ("!", "~"):
        return "%s (%s)" % (op, to_cpp_expr(args[0], libs))
    elif op == 'str':
        libs.add(CPPLibs.string)
        return "to_string(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'len':
        return "(%s).size()" % to_cpp_expr(args[0], libs)
    elif op in ('sqrt', 'log', 'sin', 'cos', 'round', 'floor',
                'ceil', 'abs'):
        libs.add(CPPLibs.math)
        return "%s(%s)" % (op, to_cpp_expr(args[0], libs))
    elif op in ('atan2', 'pow', 'min', 'max'):
        pass
    elif op == 'clear':
        return "(%s)->clear()" % to_cpp_expr(args[0], libs)
    elif op == 'reverse':
        return "reverse(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'lower':
        libs.add(CPPLibs.locale)
        return "tolower(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'upper':
        libs.add(CPPLibs.locale)
        return "toupper(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'sort':
        libs.add(CPPLibs.special)
        return "sort(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'sort_cmp':
        # We need to expand UAST syntax to include sort_cmp. It is not used in the test set though.
        libs.add(CPPLibs.special)
        return "sort_cmp(%s, &%s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'fill':
        libs.add(CPPLibs.special)
        return "fill(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'copy_range':
        libs.add(CPPLibs.special)
        return "copy_range(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'array_index':
        return "(%s)->at(%s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'contains':
        return "({container})->find({element})!=({container})->end()".format(container=to_cpp_expr(args[0], libs),
                                                                             element=to_cpp_expr(args[1], libs))
    elif op == 'string_find':
        libs.add(CPPLibs.special)
        return "string_find(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'string_find_last':
        libs.add(CPPLibs.special)
        return "string_find_last(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'string_replace_one':
        libs.add(CPPLibs.special)
        return "string_replace_one(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'string_replace_all':
        libs.add(CPPLibs.special)
        return "string_replace_all(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'concat':
        libs.add(CPPLibs.special)
        return "concat(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_concat':
        libs.add(CPPLibs.special)
        return "array_concat(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'string_insert':
        libs.add(CPPLibs.special)
        return "string_insert(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'string_split':
        libs.add(CPPLibs.special)
        return "string_split(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'string_trim':
        libs.add(CPPLibs.special)
        return "string_trim(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'substring':
        libs.add(CPPLibs.special)
        return "substring(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'substring_end':
        libs.add(CPPLibs.special)
        return "substring_end(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_push':
        libs.add(CPPLibs.special)
        return "array_push(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_pop':
        libs.add(CPPLibs.special)
        return "array_pop(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'array_insert':
        libs.add(CPPLibs.special)
        return "array_insert(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'array_remove_idx':
        libs.add(CPPLibs.special)
        return "array_remove_idx(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_remove_value':
        libs.add(CPPLibs.special)
        return "array_remove_value(%s, %s)" % to_cpp_args(libs, args[0], args[1])


