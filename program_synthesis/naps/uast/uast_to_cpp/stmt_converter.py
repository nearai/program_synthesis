from program_synthesis.naps.uast.uast_to_cpp.expr_converter import to_cpp_expr, to_cpp_type
from program_synthesis.naps.uast.uast_to_cpp.libs_to_include import CPPLibs
from program_synthesis.naps.uast import uast


def to_cpp_record_ctor_block(func, libs):
    block = func[5]
    res = []
    for v in func[4]:
        res.append("%s %s;" % (to_cpp_type(v[1], libs), v[2]))
    for stmt in block:
        if stmt[0] != 'return':
            s = to_cpp_stmt(stmt, libs)
            if s:
                res.append(s)
    return "\n".join(res)


def to_cpp_block(block, libs):
    res = []
    for stmt in block:
        s = to_cpp_stmt(stmt, libs)
        if s:
            res.append(s)
    return "\n".join(res)


# TODO: Take care of cases when the variable used in foreach is used before or after the loop.
def to_cpp_stmt(stmt, libs, semicolon=True):
    op = stmt[0]
    if op == 'if':
        if stmt[4]:
            return """
            if (%s) {
            %s
            } else {
            %s
            }
            """ % (to_cpp_expr(stmt[2], libs), to_cpp_block(stmt[3], libs), to_cpp_block(stmt[4], libs))
        else:
            return """
            if (%s) {
            %s
            }
            """ % (to_cpp_expr(stmt[2], libs), to_cpp_block(stmt[3], libs))
    elif op == 'foreach':
        it = stmt[3]
        if it[1] != 'char*' and (uast.is_set_type(it[1]) or uast.is_array(it[1])):
            return """
                    for(auto %s : *(%s)) {
                    %s
                    }
                    """ % (to_cpp_expr(stmt[2], libs), to_cpp_expr(it, libs), to_cpp_block(stmt[4], libs))
        if uast.is_map_type(it[1]):
            libs.add(CPPLibs.special)
            return """
                    for(auto %s : *map_keys(%s)) {
                    %s
                    }
                    """ % (to_cpp_expr(stmt[2], libs), to_cpp_expr(it, libs), to_cpp_block(stmt[4], libs))
        return """
        for(auto %s : %s) {
        %s
        }
        """ % (to_cpp_expr(stmt[2], libs), to_cpp_expr(it, libs), to_cpp_block(stmt[4], libs))
    elif op == 'while':
        if stmt[4]:  # Increment block is present.
            increment_block = [to_cpp_stmt(sub, libs, semicolon=False) for sub in stmt[4]]
            increment_block = ', '.join(increment_block)
            return """
            for(; %s; %s) {
            %s
            }
            """ % (to_cpp_expr(stmt[2], libs), increment_block, to_cpp_block(stmt[3], libs))
        else:
            return """
            while(%s) {
            %s
            }
            """ % (to_cpp_expr(stmt[2], libs), to_cpp_block(stmt[3], libs))
    elif op == 'break':
        return "break" + (";" if semicolon else "")
    elif op == 'continue':
        return "continue" + (";" if semicolon else "")
    elif op == 'return':
        if len(stmt[2]) > 1 and stmt[2][1] == 'void':
            return "return" + (";" if semicolon else "")
        return "return %s" % to_cpp_expr(stmt[2], libs) + (";" if semicolon else "")
    elif op == 'noop':
        return ''
    else:
        return to_cpp_expr(stmt, libs) + (";" if semicolon else "")
