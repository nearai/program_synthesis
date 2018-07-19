from program_synthesis.naps.uast.uast_to_cpp.type_converter import to_cpp_type


def get_func_decl(uast_func, libs):
    cpp_return_type = to_cpp_type(uast_func[1], libs)
    name = uast_func[2]
    arguments = []
    for a in uast_func[3]:
        arguments.append("%s %s" % (to_cpp_type(a[1], libs), a[2]))
    arguments = ", ".join(arguments)
    return "{cpp_return_type} {name}({arguments)".format(cpp_return_type=cpp_return_type,
                                                         name=name,
                                                         arguments=arguments)


def get_class_decl(uast_record, libs):
    _ = libs
    return "struct %s" % uast_record[1]
