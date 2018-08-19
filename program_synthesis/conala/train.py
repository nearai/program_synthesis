import os
import json
import collections

import tqdm

import torch

from program_synthesis.common.tools import saver

from program_synthesis.naps.pipes.pipe import Pipe
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.basic_pipes import JsonLoader, RandomAccessFile, Cycle, Merge, Batch, DropKeys

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import dataset
from program_synthesis.algolisp import arguments
from program_synthesis.algolisp.models import seq2seq_model


BASE_PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_PATH, "../../data/conala/")
CONALA_TRAIN = os.path.join(DATA_FOLDER, "conala-train.json")
CONALA_WORD_VOCAB = os.path.join(DATA_FOLDER, 'word.vocab')
CONALA_CODE_VOCAB = os.path.join(DATA_FOLDER, 'code.vocab')


class OpenJsonFile(Pipe):

    def __init__(self, filename):
        self.filename = filename

    def enter(self):
        self.data = json.load(open(self.filename))

    def exit(self):
        self.data = None

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ToCodeExample(Pipe):

    def _process(self, example):
        return dataset.CodeExample(
            text=data.tokenize_text_line(
                example['rewritten_intent'] or example['intent']),
            schema=None,
            input_tests=[],
            code_sequence=data.tokenize_code_line(example['snippet']),
            code_tree=None,
            tests=[]
        )

    def __iter__(self):
        for example in self.input:
            yield self._process(example)

    def __getitem__(self, index):
        return self._process(self.input[index])

    def __len__(self):
        return len(self.input)


def read_conala(batch_size=100, num_epochs=300):
    train = Compose([
        OpenJsonFile(CONALA_TRAIN),
        ToCodeExample(),
        Cycle(shuffle=True, times=num_epochs),
        Batch(batch_size=batch_size),
    ])
    return train
    

def create_vocabs(text_vocab_filepath, code_vocab_filepath, min_occurencies=50):
    ds = Compose([
        OpenJsonFile(CONALA_TRAIN), 
        ToCodeExample()])
    words, codes = collections.Counter(), collections.Counter()
    with ds:
        for example in ds:
            for word in example.text:
                words[word] += 1
            for token in example.code_sequence:
                codes[token] += 1

    def f(l): return sorted(k for k, v in l.items() if v >= min_occurencies)
    text_vocab = f(words)
    code_vocab = f(codes)
    def dump_to_file(filepath, vocab):
        with open(filepath, "w") as f:
            f.write("<S>\n</S>\n<UNK>\n|||\n")
            f.write("\n".join(vocab))
    dump_to_file(text_vocab_filepath, text_vocab)
    dump_to_file(code_vocab_filepath, code_vocab)


def main(args):
    if not os.path.exists(CONALA_WORD_VOCAB):
        create_vocabs(CONALA_WORD_VOCAB, CONALA_CODE_VOCAB)
    train = read_conala()
    args.word_vocab = CONALA_WORD_VOCAB
    args.code_vocab = CONALA_CODE_VOCAB
    args.vocab_mapping = False
    model = seq2seq_model.Seq2SeqModel(args)
    with train, tqdm.tqdm(smoothing=0.1) as pbar:
        for step, batch in enumerate(train):
            metrics = model.train(batch)
            pbar.update(1)
            if step % args.log_interval == 0:
                pbar.write("loss: {loss:.6f};\tlr: {lr:.8f}".format(**metrics))


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Training AlgoLisp', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    main(args)
