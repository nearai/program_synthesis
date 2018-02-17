import Levenshtein

def compute_edit_ops(source_seq, target_seq, stoi):
    source_str = u''.join(unichr(stoi(t)) for t in source_seq)
    target_str = u''.join(unichr(stoi(t)) for t in target_seq)

    ops = Levenshtein.editops(source_str, target_str)
    i, op_idx = 0, 0
    while i < len(source_seq) or op_idx < len(ops):
        if op_idx == len(ops) or i < ops[op_idx][1]:
            yield (i, 'keep', None)
            i += 1
            continue
        op_type, source_pos, target_pos = ops[op_idx]
        op_idx += 1
        if op_type == 'insert':
            yield (i, 'insert', target_seq[target_pos])
            continue
        elif op_type == 'replace':
            yield (i, 'replace', target_seq[target_pos])
        elif op_type == 'delete':
            yield (i, 'delete', None)
        else:
            raise ValueError(op_type)
        i += 1


def apply_edit_ops(source_seq, ops):
    last_i = None
    counter = 0
    for i, op, value in ops:
        assert counter == i
        if last_i is not None:
            assert last_i <= i
        if op == 'keep':
            yield source_seq[counter]
            counter += 1
        elif op == 'replace':
            yield value
            counter += 1
        elif op == 'delete':
            counter += 1
        elif op == 'insert':
            yield value
    assert counter == len(source_seq) 
