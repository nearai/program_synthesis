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
        if inst == "__globals__":
            return "__globals__%s" % expr[3]
        else:
            return "(%s)->%s" % (inst, expr[3])
    elif car == 'val':
        if expr[1] == 'real':
            return 'static_cast<double>(%s)' % expr[2]
        elif expr[1] == 'char*':
            libs.add(CPPLibs.string)
            return 'static_cast<string>(R"(%s)")' % expr[2]
        elif expr[1] == 'int':
            return "%sL" % expr[2]
        elif expr[1] == 'bool':
            return "true" if str(expr[2]) == "True" else "false"
        elif (uast.is_array(expr[1]) or uast.is_set_type(expr[1]) or uast.is_map_type(expr[1]) or
              uast.is_record_type(expr[1])) and expr[2] is None:
            return "make_shared<%s >()" % to_cpp_type(expr[1], libs, wrap_shared=False)
        return str(expr[2])
    elif car == '?:':
        expr1 = to_cpp_expr(expr[2], libs)
        expr2 = to_cpp_expr(expr[3], libs)
        expr3 = to_cpp_expr(expr[4], libs)
        return "(%s)?(%s):(%s)" % (expr1, expr2, expr3)
    elif car == 'cast':
        if expr[1] == 'char*':
            libs.add(CPPLibs.string)
            return "to_string(%s)" % to_cpp_expr(expr[2], libs)
        if expr[2][1] == 'char*' and expr[1] in ('int', 'char'):
            # This is supposed to be a string of length 1.
            return "static_cast< %s > ((%s).at(0))" % (to_cpp_type(expr[1], libs), to_cpp_expr(expr[2], libs))
        return "static_cast< %s > (%s)" % (to_cpp_type(expr[1], libs), to_cpp_expr(expr[2], libs))
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
        libs.add(CPPLibs.special)
        return "to_string(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'len':
        libs.add(CPPLibs.special)
        return "size(%s)" % to_cpp_expr(args[0], libs)
    elif op in ('sqrt', 'log', 'sin', 'cos', 'round', 'floor',
                'ceil', 'abs'):
        if op == 'abs':
            libs.add(CPPLibs.cstdlib)
        else:
            libs.add(CPPLibs.cmath)
        return "%s(%s)" % (op, to_cpp_expr(args[0], libs))
    elif op in ('atan2', 'pow', 'min', 'max'):
        if op in ('min', 'max', 'pow'):
            libs.add(CPPLibs.special)
            return "%s_(%s, %s)" % (op, to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))
        else:
            libs.add(CPPLibs.cmath)
            return "%s(%s, %s)" % (op, to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))
    elif op == 'clear':
        return "(%s)->clear()" % to_cpp_expr(args[0], libs)
    elif op == 'reverse':
        libs.add(CPPLibs.special)
        return "reverse(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'lower':
        libs.add(CPPLibs.special)
        return "tolower(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'upper':
        libs.add(CPPLibs.special)
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
        if args[0][1] == 'char*':
            return "(%s).at(%s)" % to_cpp_args(libs, args[0], args[1])
        else:
            return "(%s)->at(%s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'contains':
        libs.add(CPPLibs.special)
        return "contains(%s, %s)" % (to_cpp_expr(args[0], libs), to_cpp_expr(args[1], libs))
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
    elif op == 'array_find':
        libs.add(CPPLibs.special)
        return "array_find(%s, %s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_find_next':
        libs.add(CPPLibs.special)
        return "array_find_next(%s, %s, %s)" % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'set_push':
        return "(%s)->insert(%s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'set_remove':
        return "(%s)->erase(%s)" % to_cpp_args(libs, args[0], args[1])
    elif op == 'map_has_key':
        return '({container})->find({key})!=({container})->end()'.format(container=to_cpp_expr(args[0], libs),
                                                                         key=to_cpp_expr(args[1], libs))
    elif op == 'map_put':
        return '(%s)->at(%s)=(%s)' % to_cpp_args(libs, args[0], args[1], args[2])
    elif op == 'map_get':
        return '(%s)->at(%s)' % to_cpp_args(libs, args[0], args[1])
    elif op == 'map_keys':
        libs.add(CPPLibs.special)
        return "map_keys(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'map_values':
        libs.add(CPPLibs.special)
        return "map_values(%s)" % to_cpp_expr(args[0], libs)
    elif op == 'map_remove_key':
        return '(%s)->erase(%s)' % to_cpp_args(libs, args[0], args[1])
    elif op == 'array_initializer':
        libs.add(CPPLibs.special)
        return 'array_initializer({%s})' % (', '.join(to_cpp_expr(arg, libs) for arg in args))
    elif op == '_ctor':
        if expr[1] == "char*":
            return '""'
        if expr[1] == 'int':
            return "0"
        if (uast.is_array(expr[1]) or uast.is_set_type(expr[1])) and len(args) == 1 and (
                uast.is_array(args[0][1]) or uast.is_set_type(args[0][1])):
            # This is the case when one array/set is constructed from elements in another array set.
            libs.add(CPPLibs.special)
            return "from_iterable<%s >(%s)" % (to_cpp_type(expr[1], libs, wrap_shared=False),
                                               to_cpp_expr(args[0], libs))
        if uast.is_array(expr[1]) and args:
            # We need to initialize a large n-dimensional array and fill it with default values.
            libs.add(CPPLibs.special)
            return "move(init_darray<%s > (%s))" % (
                to_cpp_type(expr[1], libs, wrap_shared=False),
                ', '.join(to_cpp_expr(arg, libs) for arg in args))
        return 'make_shared<%s >(%s)' % (to_cpp_type(expr[1], libs, wrap_shared=False),
                                         ', '.join(to_cpp_expr(arg, libs) for arg in args))
    else:
        return '%s(%s)' % (op, ', '.join(to_cpp_expr(arg, libs) for arg in args))
