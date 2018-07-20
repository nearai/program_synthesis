import six
import unicodedata


PAD_TOKEN = -1
GO_TOKEN_STR = "<S>"
GO_TOKEN = 0
END_TOKEN_STR = "</S>"
END_TOKEN = 1
UNK_TOKEN_STR = "<UNK>"
UNK_TOKEN = 2
SEQ_SPLIT_STR = "|||"
SEQ_SPLIT = 3


def replace_pad_with_end(var):
    new_var = var.clone()
    new_var[var == PAD_TOKEN] = END_TOKEN
    return new_var


def _decode(line):
    if six.PY2:
        return line.decode('utf-8')
    return line


def _encode(line):
    if six.PY2:
        return line.encode('utf-8')
    return line


def load_vocab(filename, mapping=True):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            if mapping:
                try:
                    key, value = _decode(line).strip().split('\t')
                    if key in vocab:
                        raise ValueError(
                            "Got %s key again with value %s" % (key, value))
                    vocab[key] = int(value)
                except ValueError as e:
                    print(line)
                    raise e
            else:
                key = _decode(line).strip()
                if key in vocab:
                    raise ValueError(
                        "Got %s key again with value %s" % (key, idx))
                vocab[key] = idx
    print("Loaded vocab %s: %d" % (filename, len(vocab)))
    return vocab


def tokenize_code_line(line):
    if six.PY2:
        if isinstance(line, str):
            line = unicode(line)
    def is_prefix(prev_tokens, new_token):
        gonna = prev_tokens + new_token
        if gonna in ('++', '--', '!=', '==', '<<', '>>', '&&', '||', '>=', '<='):
            return True
        return False
    tokens, last_token = [], ''
    new_type, prev_type = None, None
    push = False
    for i in range(len(line)):
        if unicodedata.category(line[i]) in ('Zs', 'Cc', 'Cf'):
            new_type = 'W'
            push = True
        elif unicodedata.category(line[i])[0] == 'L':
            new_type = 'L'
            if prev_type not in ('D', 'L') and not (prev_type == 'P' and last_token[-1] == '_'):
                push = True
        elif unicodedata.category(line[i])[0] == 'N':
            new_type = 'D'
            if prev_type not in ('D', 'L') and not (prev_type == 'P' and last_token[-1] == '_'):
                push = True
            if prev_type == 'P' and last_token[-1] == '.':
                push = False
            if last_token == '-':
                push = False
        elif unicodedata.category(line[i])[0] in ('P', 'S'):
            new_type = 'P'
            if prev_type != 'P' or not is_prefix(last_token, line[i]):
                push = True
            if line[i] == '_' and (prev_type in ('D', 'L') or prev_type == 'P' and last_token[-1] == '_'):
                push = False
            if i + 1 < len(line) and line[i] == '.' and prev_type == 'D' and unicodedata.category(line[i + 1])[0] == 'N':
                push = False
        else:
            print("WTF:", line[i], unicodedata.category(line[i]))
        if push:
            if last_token:
                tokens.append(last_token)
            last_token = ""
        if new_type != 'W':
            last_token += line[i]
        prev_type = new_type
        push = False
    if last_token:
        tokens.append(last_token)
    return tokens