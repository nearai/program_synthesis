import collections
import random
import time
import re
import tokenize
import json
import string
import unicodedata
import os
import sys
import six

from IPython.lib import pretty


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


def load_vocab(filename, mapping=True):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            if mapping:
                try:
                    if six.PY2:
                        key, value = line.decode('utf-8').strip().split('\t')
                    else:
                        key, value = line.strip().split('\t')
                    if key in vocab:
                        raise ValueError(
                            "Got %s key again with value %s" % (key, value))
                    vocab[key] = int(value)
                except ValueError as e:
                    print(line)
                    raise e
            else:
                if six.PY2:
                    key = line.decode('utf-8').strip()
                else:
                    key = line.strip()
                if key in vocab:
                    raise ValueError(
                        "Got %s key again with value %s" % (key, idx))
                vocab[key] = idx
    print("Loaded vocab %s: %d" % (filename, len(vocab)))
    return vocab


def save_vocab(filename, vocab):
    with open(filename, 'w') as f:
        for key, value in sorted(vocab.items(), key=lambda x: x[1]):
            line = u'%s\t%d\n' % (key, value)
            if six.PY2:
                line = line.encode('utf-8')
            f.write(line)


def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}


def get_vocab(freqs, threshold):
    vocab = {GO_TOKEN_STR: GO_TOKEN, END_TOKEN_STR: END_TOKEN,
             UNK_TOKEN_STR: UNK_TOKEN, SEQ_SPLIT_STR: SEQ_SPLIT}
    init_size = len(vocab)
    for idx, (key, value) in enumerate(
            sorted(freqs.items(), key=lambda x: x[1], reverse=True)):
        if value < threshold:
            break
        vocab[key] = idx + init_size
    return vocab


class PlaceholderVocab(object):

    def __init__(self, vocab, num_placeholders):
        self._vocab = vocab
        self._rev_vocab = get_rev_vocab(vocab)
        self._num_placeholders = num_placeholders
        self.reset()

    def reset(self):
        self._last_idx = 0
        self._indicies = list(range(len(self._vocab), len(self._vocab) + self._num_placeholders))
        random.shuffle(self._indicies)
        self._placeholders = {}
        self._rev_placeholders = {}

    def stoi(self, key):
        if key in self._vocab:
            return self._vocab[key]
        else:
            if key not in self._placeholders:
                if self._last_idx >= self._num_placeholders:
                    return UNK_TOKEN
                self._placeholders[key] = self._indicies[self._last_idx]
                self._rev_placeholders[self._indicies[self._last_idx]] = key
                self._last_idx += 1
            return self._placeholders[key]

    def itos(self, idx):
        if idx in self._rev_vocab:
            return self._rev_vocab[idx]
        else:
            return self._rev_placeholders.get(idx, "PL@%d" % idx)

    def __len__(self):
        return len(self._vocab) + self._num_placeholders


class Vocab(object):
    def __init__(self, keys):
        self.keys = keys
        self.key_to_idx = {key: i for i, key in enumerate(keys)}

    def stoi(self, key):
        return self.key_to_idx[key]

    def itos(self, idx):
        return self.keys[idx]

    def __len__(self):
        return len(self.keys)


def flatten_lisp_code(code):
    if isinstance(code, basestring):
        return [code]
    elif isinstance(code, (int, float)):
        return [str(code)]
    elif isinstance(code, list):
        res = ["("]
        for elem in code:
            res.extend(flatten_lisp_code(elem))
        res.append(")")
        return res


def flatten_code(code, language):
    if language == 'lisp':
        return flatten_lisp_code(code)
    else:
        raise ValueError("Unknown language: %s" % language)


def is_flat_code(seq):
    if len(seq) > 0 and seq[0] == "(" and seq[-1] == ")":
        return True
    return False


def unflatten_lisp_code(seq):
    def _unflatten(idx):
        if seq[idx] != "(":
            return seq[idx], idx + 1
        else:
            res = []
            idx += 1
            while seq[idx] != ")":
                elem, idx = _unflatten(idx)
                res.append(elem)
            return res, idx + 1
    tree, last_idx = _unflatten(0)
    return tree, last_idx == len(seq)


def unflatten_code(seq, language):
    if language == 'lisp':
        return unflatten_lisp_code(seq)
    else:
        raise ValueError("Unknown language: %s" % language)


class SExprWrapper(object):
    def __init__(self, obj):
        self.obj = obj

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('...')
            return

        if isinstance(self.obj, (list, tuple)):
            with p.group(2, '(', ')'):
                for i, item in enumerate(self.obj):
                    if i:
                        p.breakable()
                    p.pretty(SExprWrapper(item))
        elif isinstance(self.obj, (str, unicode)):
            p.text(self.obj)
        else:
            p.pretty(self.obj)


def format_code(code, language):
    if language == 'lisp':
        return pretty.pretty(SExprWrapper(code))
    else:
        raise NotImplementedError(language)


def tokenize_text_line(line, skip_tokens=u'`'):
    if isinstance(line, str):
        line = unicode(line)
    tokens, last_token = [], ''
    new_type, prev_type = None, None
    push = False
    for i in range(len(line)):
        if unicodedata.category(line[i]) in ('Zs', 'Cc', 'Cf'):
            new_type = 'W'
            push = True
        elif unicodedata.category(line[i])[0] in ('L', 'N'):
            new_type = 'L'
            if prev_type != 'L':
                push = True
        elif unicodedata.category(line[i])[0] in ('P', 'S'):
            new_type = 'P'
            push = True
        else:
            print("WTF:", line[i], unicodedata.category(line[i]))
        if push:
            if last_token and last_token not in skip_tokens:
                tokens.append(last_token.lower())
            last_token = ""
        if new_type != 'W':
            last_token += line[i]
        prev_type = new_type
        push = False
    if last_token and last_token not in skip_tokens:
        tokens.append(last_token.lower())
    return tokens


def tokenize_code_line(line):
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
