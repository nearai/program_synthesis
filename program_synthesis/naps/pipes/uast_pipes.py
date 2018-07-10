import random

from program_synthesis.naps.uast.lisp_to_uast import lisp_to_uast, uast_to_lisp
from .pipe import Pipe


class SelectPseudocode(Pipe):
    def __init__(self, texts_key, text_key):
        self.texts_key = texts_key
        self.text_key = text_key

    def __iter__(self):
        for d in self.input:
            yield {**d, **{self.text_key: random.choice(d[self.texts_key])}}
        return


class SkipPartial(Pipe):
    def __init__(self, is_partial_key):
        self.is_partial_key = is_partial_key

    def __iter__(self):
        for d in self.input:
            if not d.get(self.is_partial_key, False):
                yield d
        return


class ShuffleVariables(Pipe):
    def __init__(self, code_tree_key, code_sequence_key, text_key):
        self.code_tree_key = code_tree_key
        self.code_sequence_key = code_sequence_key
        self.text_key = text_key

    def make_remap(self, names, prefix, upto):
        cur = list(names[prefix].values())
        values = ['%s%d' % (prefix, i) for i in range(upto)]
        random.shuffle(values)
        return dict(zip(cur, values))

    def __iter__(self):
        for d in self.input:
            names = {'struct': {}, 'func': {}, 'var': {}}
            uast_to_lisp.remap_uast(d[self.code_tree_key], names)
            remap = self.make_remap(names, 'var', 35)
            remap.update(self.make_remap(names, 'func', 7))
            remap.update(self.make_remap(names, 'struct', 2))
            new_text = [remap.get(word, word) for word in d[self.text_key]]
            new_code_sequence = [remap.get(token, token) for token in d[self.code_sequence_key]]
            new_code_tree = lisp_to_uast(new_code_sequence)
            yield {**d, **{self.text_key: new_text,
                           self.code_sequence_key: new_code_sequence,
                           self.code_tree_key: new_code_tree}}
        return