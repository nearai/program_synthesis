from program_synthesis.naps.uast.uast_to_cpp.expr_converter import to_cpp_expr


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
        return """
        foreach(auto %s : %s) {
        %s
        }
        """ % (to_cpp_expr(stmt[2], libs), to_cpp_expr(stmt[3], libs), to_cpp_block(stmt[4], libs))
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
        return "return %s" % to_cpp_expr(stmt[2], libs) + (";" if semicolon else "")
    elif op == 'noop':
        return ''
    else:
        return to_cpp_expr(stmt, libs) + (";" if semicolon else "")
