from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import CPPLibs
from program_synthesis.naps.uast.uast_to_cpp.decl_extractor import get_class_decl, get_func_decl, get_ctor_decl
from program_synthesis.naps.uast.uast_to_cpp.type_converter import to_cpp_type
from program_synthesis.naps.uast.uast_to_cpp.stmt_converter import to_cpp_block
from program_synthesis.naps.uast import uast


def convert_libs(libs):
    if not libs:
        return ""
    if CPPLibs.special in libs:
        return "#include <special_lib.h>\nusing namespace std;"
    result = []
    for lib in libs:
        result.append("#include<%s>" % str(lib).split('.')[1])
    result.append('using namespace std;')
    return "\n".join(result)


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
                           ctor_body=func_body_to_cpp(func, libs)))
                break
        result += """{decl} {{
        {body}
        }};
        """.format(decl=get_class_decl(record, libs), body=body)
    result += "\n".join(ctors)
    return result


def program_to_cpp(code_tree):
    result = ""
    libs = set()
    if len(code_tree['types']) > 1:
        # Forward declarations might be needed.
        for record in code_tree['types']:
            result += get_class_decl(record, libs) + ";\n"
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

    result = convert_libs(libs) + result
    return result, CPPLibs.special in libs
