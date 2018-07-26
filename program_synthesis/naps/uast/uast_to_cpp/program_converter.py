"""Converts UAST program to C++.
Note, we might later need to replace all string's with wstring's and all char's with wchar_t's, see:
https://stackoverflow.com/a/402918
"""

import copy
from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import convert_libs
from program_synthesis.naps.uast.uast_to_cpp.decl_extractor import get_class_decl, get_func_decl, get_ctor_decl
from program_synthesis.naps.uast.uast_to_cpp.type_converter import to_cpp_type
from program_synthesis.naps.uast.uast_to_cpp.stmt_converter import to_cpp_block, to_cpp_record_ctor_block, to_cpp_stmt
from program_synthesis.naps.uast.uast_to_cpp.expr_converter import to_cpp_expr
from program_synthesis.naps.uast import uast


def func_body_to_cpp(func, libs):
    result = []
    for v in func[4]:
        result.append("%s %s;" % (to_cpp_type(v[1], libs), v[2]))
    result.append(to_cpp_block(func[5], libs))
    return "\n".join(result)


def records_to_cpp(code_tree, libs):
    result = ""
    # Postpone ctor definitions.
    ctors = []
    for record in code_tree['types']:
        body = []
        for field_var in record[2].values():
            body.append("%s %s;" % (to_cpp_type(field_var[1], libs), field_var[2]))
        body = ";\n".join(body)
        for func in code_tree['funcs']:
            if func[0] == 'ctor' and uast.is_record_type(func[1]) and uast.type_to_record_name(func[1]) == record[1]:
                body += "\n" + get_ctor_decl(func, libs) + ";"
                ctors.append("""
                {definition} {{
                {ctor_body}
                }}
                """.format(definition=get_ctor_decl(func, libs, definition=True),
                           ctor_body=to_cpp_record_ctor_block(func, libs)))
                ctors.append("""
                {definition} {{
                    return make_shared<{type} >({args});
                }}
                """.format(definition=get_func_decl(func, libs), type=to_cpp_type(func[1], libs, wrap_shared=False),
                           args=', '.join(to_cpp_expr(arg, libs) for arg in func[3])))
                break
        result += """{decl} {{
        {body}
        }};
        """.format(decl=get_class_decl(record, libs), body=body)
    result += "\n".join(ctors)
    return result


def rename_this(code_tree):
    # For this functions that use this as an argument rename it to this_.
    def recursively_rename(node):
        if not isinstance(node, list) or not node:
            return
        if node[0] == 'var' and node[2] == 'this':
            node[2] = 'this_'
            return
        for el in node:
            recursively_rename(el)

    for func in code_tree['funcs']:
        if func[3] and func[3][0][2] == 'this':
            func[3][0][2] = 'this_'
            recursively_rename(func[5])
        elif func[0] != 'ctor' and func[4] and func[4][0][2] == 'this':
            func[4][0][2] = 'this_'
            recursively_rename(func[5])


def try_repair_types(code_tree):
    # Try repairing types by transplanting from siblings.
    def recursively_repair(node, parent):
        if not isinstance(node, list) or not node:
            return
        if len(node) > 1 and node[1] == '?' and len(parent) > 1:
            for sibling in parent:
                if isinstance(sibling, list) and len(sibling) > 1 and sibling != node:
                    node[1] = sibling[1]
                return
        for el in node:
            recursively_repair(el, node)
    for f in code_tree["funcs"]:
        recursively_repair(f[5], f)


def program_to_cpp(code_tree):
    result = ""
    header_result = ""
    libs = set()
    # Copy functions and records because we will prune static variables.
    code_tree = copy.deepcopy(code_tree)
    try_repair_types(code_tree)
    rename_this(code_tree)
    # Forward declarations of types always go first.
    for record in code_tree['types']:
        if record[1] != '__globals':
            result += get_class_decl(record, libs) + ";\n"

    # First take care of the static variables.
    if any(t[1] == '__globals__' for t in code_tree["types"]):
        for i in range(len(code_tree["types"])):
            t = code_tree["types"][i]
            if t[1] != '__globals__':
                continue
            for v in t[2].values():
                result += 'static %s __globals__%s;\n' % (to_cpp_type(v[1], libs), v[2])
            # Remove this fake struct so that we don't have to work with it in the future.
            code_tree["types"].pop(i)
            break
    # ... and __init__ function that initializes them.
    if any(f[2] == '__globals__.__init__' for f in code_tree["funcs"]):
        for i in range(len(code_tree["funcs"])):
            f = code_tree["funcs"][i]
            if f[2] != '__globals__.__init__':
                continue
            global_init_body = []
            for stmt in f[5]:
                if (stmt[0] == 'assign' and stmt[2][0] == 'field' and stmt[3] is None or
                        stmt[3][0] == 'val' and stmt[3][2] is None):
                    # Assigning None to an array, set or dict can be ignored.
                    continue
                else:
                    global_init_body.append(to_cpp_stmt(stmt, libs, semicolon=True))
            if global_init_body:
                result += """void __globals__init__() {
                    %s
                }
                """ % '\n'.join(global_init_body)
                header_result += "extern void __globals__init__();\n"
            code_tree["funcs"].pop(i)
            break

    if len([f for f in code_tree['funcs'] if f[0] == 'func']) > 1:
        # Forward declarations for functions might be needed.
        for func in code_tree['funcs']:
            if func[0] != 'func':
                continue
            result += get_func_decl(func, libs) + ";\n"

    if code_tree['types']:
        result += records_to_cpp(code_tree, libs)
    for func in code_tree['funcs']:
        if func[0] == 'ctor':
            continue
        result += """
                {definition} {{
                {ctor_body}
                }}\n
                """.format(definition=get_func_decl(func, libs),
                           ctor_body=func_body_to_cpp(func, libs))
        if func[2] == '__main__':
            header_result += "extern %s;\n" % get_func_decl(func, libs)

    result = '#include "lib_program.h"\n' + convert_libs(libs) + result
    header_result = convert_libs(libs) + header_result
    return result, header_result
