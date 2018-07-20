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
        return "sort_cmp(%s, &%s)" % (to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))
    elif op == 'fill':
        libs.add(CPPLibs.special)
        return "fill(%s, %s)" % (to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))
    elif op == 'copy_range':
        libs.add(CPPLibs.special)
        return "copy_range(%s, %s, %s)" % (to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs),
                                           to_cpp_expr(args[2], libs))
    elif op == 'array_index':
        libs.add(CPPLibs.vector)
        return "(%s)->at(%s)" % (to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))



