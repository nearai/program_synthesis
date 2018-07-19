from program_synthesis.naps.uast import uast
from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import CPPLibs


_STANDARD_TYPES = {
    "bool": "bool",
    "char": "char",
    # Python strings are immutable therefore there is no need to wrap C++ string into shared_ptr.
    "char*": "string",
    "int": "long",
    "real": "double"
}


class UnknownUASTType(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def to_cpp_type(t, libs):
    std = _STANDARD_TYPES.get(t, None)
    if std is not None:
        if std == "string":
            libs.add(CPPLibs.string)
        return std
    elif uast.is_map_type(t):
        libs.add(CPPLibs.map)
        libs.add(CPPLibs.memory)
        t_key = uast.get_map_key_type(t)
        t_value = uast.get_map_value_type(t)
        return "shared_ptr<map< %s, %s > >" % (to_cpp_type(t_key, libs), to_cpp_type(t_value, libs))
    elif uast.is_set_type(t):
        libs.add(CPPLibs.set)
        libs.add(CPPLibs.memory)
        t_sub = uast.get_set_subtype(t)
        return "shared_ptr<set< %s > >" % to_cpp_type(t_sub, libs)
    elif uast.is_array(t):
        libs.add(CPPLibs.vector)
        libs.add(CPPLibs.memory)
        t_sub = uast.get_array_subtype(t)
        return "shared_ptr<vector< %s > >" % to_cpp_type(t_sub, libs)
    elif uast.is_record_type(t):
        libs.add(CPPLibs.memory)
        return "shared_ptr< %s >" % uast.type_to_record_name(t)
    else:
        raise UnknownUASTType(t)
