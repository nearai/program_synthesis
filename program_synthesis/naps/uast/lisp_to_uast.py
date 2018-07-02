from . import uast, uast_to_lisp


class LispSyntaxException(Exception):
    pass


def unflatten_lisp(seq):
    def _unflatten(idx):
        if seq[idx] == '(':
            res = []
            idx += 1
            while seq[idx] != ')':
                elem, idx = _unflatten(idx)
                res.append(elem)
            return tuple(res), idx + 1
        elif seq[idx] == '[':
            res = []
            idx += 1
            while seq[idx] != ']':
                elem, idx = _unflatten(idx)
                res.append(elem)
            return res, idx + 1
        elif seq[idx] == '"':
            res = ""
            idx += 1
            while seq[idx] != '"':
                res += seq[idx]
                idx += 1
            res = uast_to_lisp.unescape_string(res)
            return '"%s"' % res, idx + 1
        else:
            res = seq[idx]
            if seq[idx + 1] in ('*', '%'):
                while seq[idx + 1] in ('*', '%'):
                    res += seq[idx + 1]
                    idx += 1
            return res, idx + 1
    try:
        tree, last_idx = _unflatten(0)
    except IndexError:
        raise LispSyntaxException()
    return tree


def _read_out_type(seq, idx):
    res = seq[idx]
    if res == "<" :
        cnt = 1
        while True:
            idx += 1
            res += seq[idx]
            if seq[idx] == "<":
                cnt += 1
            if seq[idx][0] == ">":
                cnt -=1
                if len(seq[idx]) == 2 and seq[idx][1] == ">":
                    cnt -= 1
                if cnt == 0:
                    break
    return res, idx + 1


def lisp_type_to_uast(type_):
    if isinstance(type_, (tuple, list)):
        return lisp_type_to_uast(''.join(type_))
    elif type_.endswith('*'):
        return lisp_type_to_uast(type_[:-1]) + '*'
    elif type_.startswith('struct'):
        return type_ + '#'
    elif '|' in type_ and type_[0] == '<' and type_[-1] == '>':  # First look for substring as a minor optimization.
        key_type = uast.get_map_key_type(type_)
        value_type = uast.get_map_value_type(type_)
        return "<%s|%s>" % (lisp_type_to_uast(key_type), lisp_type_to_uast(value_type))
    return type_


def lisp_struct_to_uast(seq, type_map):
    fields = {}
    struct_name = seq[1]
    for field in seq[2]:
        name = field[0]
        type_ = lisp_type_to_uast(field[1:])
        fields[name] = ['var', type_, name]
        type_map[struct_name + '#.' + name] = type_
    return ['struct', struct_name, fields]


def prepare_func(seq, type_map):
    func_name = seq[1]
    if func_name == '__globals_____init__':
        func_name = '__globals__.__init__'
    return_type = lisp_type_to_uast(seq[2])
    type_map[func_name] = return_type
    return seq[0], return_type, func_name, seq[3], seq[4], seq[5]


def lisp_func_to_uast(kind, return_type, func_name, arg_lst, var_lst, body_lst, type_map):
    args = [['var', lisp_type_to_uast(item[1:]), item[0]] for item in arg_lst]
    variables = [['var', lisp_type_to_uast(item[1:]), item[0]] for item in var_lst]
    local_type_map = {k: v for k, v in type_map.items()}
    for _, type_, name in args + variables:
        local_type_map[name] = type_
    if kind == 'ctor':
        local_type_map['this'] = return_type
    body = lisp_block_to_uast(body_lst, local_type_map, return_type)
    return [kind, return_type, func_name, args, variables, body]


def can_be_float(f):
    try:
        float(f)
        return True
    except:
        return False


def func_type_inference(name, args, type_map, default):
    func = uast.DEFAULT_TYPE_FUNCS.get(name, None)
    if func is not None:
        return func(*[x[1] for x in args])
    else:
        type_ = type_map.get(name, default)
        # assert type_ != "?", (name, args, type_map, default)
        return type_


def lisp_expression_to_uast(seq, type_map, default='?'):
    if isinstance(seq, tuple):
        if seq[0] == '=':
            left = lisp_expression_to_uast(seq[1], type_map)
            right = lisp_expression_to_uast(seq[2], type_map, default=left[1])
            return ['assign', left[1], left, right]
        if seq[0] == 'cast':
            return ['cast', seq[1], lisp_expression_to_uast(seq[2], type_map)]
        if seq[0] == '?':
            choices = [lisp_expression_to_uast(s, type_map) for s in seq[2:]]
            type_ = uast.arithmetic_op_type(choices[0][1], choices[1][1], allow_same=True)
            return ['?:', type_, lisp_expression_to_uast(seq[1], type_map, 'bool')] + choices
        if seq[0] == '_ctor':
            tp, idx = _read_out_type(seq, 1)
            args = [lisp_expression_to_uast(x, type_map) for x in seq[idx:]]
            return ['invoke', tp, '_ctor', args]
        if seq[0] == 'field':
            left = lisp_expression_to_uast(seq[1], type_map)
            right_type = type_map.get(left[1] + '.' + seq[2], default)
            return ['field', right_type, left, seq[2]]
        if seq[0].startswith('func') or seq[0] in uast_to_lisp._DEFAULT_NAMES['func']:
            args = [lisp_expression_to_uast(x, type_map) for x in seq[1:]]
            return ['invoke', func_type_inference(seq[0], args, type_map, default), seq[0], args]
    else:
        if seq.startswith('var') or seq in uast_to_lisp._DEFAULT_NAMES['var']:
            return ['var', type_map.get(seq, default), seq]
        else:
            if seq == 'None':
                return ['val', default, None]
            elif seq in ('True', 'False'):
                return ['val', 'bool', seq == 'True']
            elif seq[0] == '"':
                return ['val', 'char*', seq[1:-1]]

            try:
                return ['val', 'int', int(seq)]
            except ValueError:
                try:
                    if '_' in seq:
                        return ['val', 'real', float(seq.replace('__', '+').replace('_', '-'))]
                    return ['val', 'real', float(seq)]
                except ValueError:
                    return ['val', 'char*', seq]
    return seq


def lisp_statement_to_uast(seq, type_map, return_expected_type=None):
    if seq[0] == 'if':
        return ['if', 'void', 
                lisp_expression_to_uast(seq[1], type_map, default='bool'),
                lisp_block_to_uast(seq[2], type_map), 
                lisp_block_to_uast(seq[3], type_map)]
    if seq[0] == 'return':
        return ['return', 'void', lisp_expression_to_uast(seq[1], type_map, return_expected_type or 'void')]
    if seq[0] == 'while':
        return ['while', 'void', 
                lisp_expression_to_uast(seq[1], type_map, default='bool'),
                lisp_block_to_uast(seq[2], type_map), 
                lisp_block_to_uast(seq[3], type_map)]
    if seq[0] == 'foreach':
        return ['foreach', 'void', 
                lisp_expression_to_uast(seq[1], type_map),
                lisp_expression_to_uast(seq[2], type_map),
                lisp_block_to_uast(seq[3], type_map)]
    if seq[0] == 'noop':
        return ['noop']
    if seq[0] in ('break', 'continue'):
        return [seq[0], 'void']
    return lisp_expression_to_uast(seq, type_map)


def lisp_block_to_uast(seq, type_map, return_expected_type=None):
    return [lisp_statement_to_uast(elem, type_map, return_expected_type) for elem in seq]


def lisp_to_uast(code_sequence):
    tree = unflatten_lisp(code_sequence)
    result = {'types': [], 'funcs': []}
    funcs = []
    type_map = {'__globals__': '__globals__#'}
    try:
        for elem in tree:
            if elem[0] == 'struct':
                result['types'].append(lisp_struct_to_uast(elem, type_map))
            elif elem[0] in ('func', 'ctor'):
                funcs.append(prepare_func(elem, type_map))
            else:
                raise LispSyntaxException("Wrong type of the top level token: %s (%s)" % (
                    elem[0], elem
                ))
        for kind, return_type, func_name, arg_lst, var_lst, body_lst in funcs:
            result['funcs'].append(lisp_func_to_uast(
                kind, return_type, func_name, arg_lst, var_lst, body_lst, type_map))
    except IndexError:
        raise LispSyntaxException()
    return result
