# The following types are used as an input: {'char***', 'real', 'char**', 'real*', 'char*', 'int', 'int**', 'int*'}
"""
Runs the program from the shared library using the input tests. If the result matches the expected output it prints
"CORRECT" into stdio otherwise it prints "INCORRECT".
"""

from program_synthesis.naps.uast.uast_to_cpp.expr_converter import to_cpp_type
from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import convert_libs


def _init_ndarray(arr, arg_type, libs):
    if not isinstance(arr, (list, tuple)):
        if arg_type == "char*":
            yield ' = R"(%s)";' % arr
        else:
            yield " = %s;" % arr
        return
    yield " = make_shared<%s >(%s);" % (to_cpp_type(arg_type, libs, wrap_shared=False), len(arr))
    for i, el in enumerate(arr):
        for sub in _init_ndarray(el, arg_type[:-1], libs):
            yield ("->at(%s)" % i) + sub
    return


def test_to_cpp(code_tree, header, test):
    func = [f for f in code_tree['funcs'] if f[2] == '__main__'][0]

    # Add variable initialization.
    libs = set()
    body = []
    to_init = [("arg%s" % arg_id, input_, arg[1])
               for arg_id, (input_, arg) in enumerate(zip(test['input'], func[3]))]
    to_init.append(("expected", test['output'], func[1]))
    for name, value, type_ in to_init:
        if type_ in ('real', 'char*', 'int'):
            # Scalar.
            if type_ == 'char*':
                value = 'R"(%s)"' % value
            body.append("%s %s = %s ;" % (to_cpp_type(type_, libs), name, value))
        else:
            # Multidimensional array.
            body.append("%s %s;" % (to_cpp_type(type_, libs), name))
            for init_element in _init_ndarray(value, type_, libs):
                body.append(name + init_element)

    # Call the actual program.
    if "__globals__init__" in header:
        body.append("__globals__init__();")
    body.append("%s actual = __main__(%s);" % (to_cpp_type(func[1], libs),
                                               ", ".join("arg%s" % i for i in range(len(func[3])))))

    # Compare the output.

    body = "\n".join(body)
    result = """
    #include "lib_program.h"
    #include "output_comparator.h"
    #include<iostream>
    
    using namespace std;
    int main() {
        %s
        if (same_output(expected, actual)) {
            cout << "CORRECT" << endl;
        } else {
            cout << "INCORRECT" << endl;
            print_output(actual);
        }
        return 0;
    }
    """ % body
    result = convert_libs(libs) + result
    return result
