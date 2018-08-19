from itertools import chain
from program_synthesis.naps.examples.seq2seq import data


class WordCodeVocab(object):
    def __init__(self, word_vocab, code_vocab):
        self.vocab = {
            data.GO_TOKEN_STR: data.GO_TOKEN,
            data.END_TOKEN_STR: data.END_TOKEN,
            data.UNK_TOKEN_STR: data.UNK_TOKEN,
            data.SEQ_SPLIT_STR: data.SEQ_SPLIT
        }
        for key in sorted(chain(word_vocab, code_vocab)):
            self.vocab.setdefault(key, len(self.vocab))

        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)


def load_vocabs(word_filename, code_filename):
    return WordCodeVocab(data.load_vocab(word_filename), data.load_vocab(code_filename))


class WordCodePlaceholdersVocab(object):
    def __init__(self, word_code_vocab, num_placeholders):
        self.word_code_vocab = word_code_vocab
        self.num_placeholders = num_placeholders
        self.placeholders = dict()
        self.rev_placeholders = dict()

    @property
    def vocab_size(self):
        return self.word_code_vocab.vocab_size + len(self.placeholders)

    def _get_or_create_placeholder(self, token):
        idx = self.placeholders.get(token)
        if idx is not None:
            return idx
        elif len(self.placeholders) < self.num_placeholders:
            new_idx = self.vocab_size
            self.placeholders[token] = new_idx
            self.rev_placeholders[new_idx] = token
            return new_idx
        else:
            return data.UNK_TOKEN

    def wordtoi(self, key):
        idx = self.word_code_vocab.vocab.get(key)
        if idx is not None:
            return idx, idx
        else:
            return self._get_or_create_placeholder(key), data.UNK_TOKEN

    def codetoi(self, key):
        idx = self.word_code_vocab.vocab.get(key)
        if idx is not None:
            return idx, idx
        else:
            return self.placeholders.get(key, data.UNK_TOKEN), data.UNK_TOKEN

    def itoword(self, idx):
        if idx < self.word_code_vocab.vocab_size:
            return self.word_code_vocab.rev_vocab[idx]
        else:
            return self.rev_placeholders[idx]

    def itocode(self, idx):
        if idx < self.word_code_vocab.vocab_size:
            return self.word_code_vocab.rev_vocab[idx]
        else:
            return self.rev_placeholders[idx]
