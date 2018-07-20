import os
import multiprocessing as mp
import tqdm

import torch

from program_synthesis.common.tools import reporter as reporter_lib
from program_synthesis.common.tools import saver
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.basic_pipes import JsonLoader, RandomAccessFile, Batch, DropKeys, KeepKeys

from program_synthesis.naps.examples.seq2seq.pointer_seq2seq_model import PointerSeq2SeqModel
from program_synthesis.naps.examples.seq2seq.pipes import Buffer, SortBatchByLen, ShuffleVariables, SkipPartial, \
    WeightedMerge, EndlessShuffleCycle, Identity, SelectPseudocode, FilterCodeLength
from program_synthesis.naps.examples.seq2seq import arguments


BASE_PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_PATH, "../../../../data/naps/")
VERSION = "1.0"
TRAIN_A_PATH = os.path.join(DATA_FOLDER, "naps.trainA.{version}.jsonl".format(version=VERSION))
TRAIN_B_PATH = os.path.join(DATA_FOLDER, "naps.trainB.{version}.jsonl".format(version=VERSION))
WORD_VOCAB_PATH = os.path.join(BASE_PATH, "vocabs/word.vocab")
CODE_VOCAB_PATH = os.path.join(BASE_PATH, "vocabs/code.vocab")


def read_naps_dataset_batched(batch_size=100, trainB_weight=1, variable_shuffle=True, dataset_filter_code_length=500):
    trainA = Compose([
        RandomAccessFile(TRAIN_A_PATH),
        JsonLoader(),
        EndlessShuffleCycle(),
        SelectPseudocode(),
        DropKeys(["texts", "is_training"])
    ])

    trainB = Compose([
        RandomAccessFile(TRAIN_B_PATH),
        JsonLoader(),
        EndlessShuffleCycle(),
        SkipPartial(is_partial_key="is_partial"),
    ])

    train = Compose([
        WeightedMerge(input_pipes=[trainA, trainB], p=[1.0, trainB_weight]),
        (ShuffleVariables(code_tree_key="code_tree", code_sequence_key="code_sequence", text_key="text")
         if variable_shuffle else Identity()),
        KeepKeys(["text", "code_sequence"]),
        FilterCodeLength(dataset_filter_code_length=dataset_filter_code_length),
        Batch(batch_size=batch_size),
        SortBatchByLen(key="text")
    ])
    return train


def train_start(args):
    print("\tModel path: %s" % args.model_dir)
    m = PointerSeq2SeqModel(args)
    m.model.train()
    saver.save_args(args)
    return m


def train(args):
    print("Training:")
    args.word_vocab = WORD_VOCAB_PATH
    args.code_vocab = CODE_VOCAB_PATH
    train_data = read_naps_dataset_batched(batch_size=args.batch_size, trainB_weight=args.trainB_weight,
                                           variable_shuffle=args.variable_shuffle,
                                           dataset_filter_code_length=args.dataset_filter_code_length)
    train_data = Buffer(train_data, max_buffer_size=10, num_workers=1)
    m = train_start(args)
    reporter = reporter_lib.Reporter(
        log_interval=args.log_interval, logdir=args.model_dir,
        smooth_interval=args.log_interval)

    m.worker_pool = mp.Pool(min(mp.cpu_count(), 7))
    with train_data, tqdm.tqdm() as pbar:
        for step, batch in enumerate(train_data):
            if step > args.num_steps:
                break
            res = m.train(batch)
            reporter.record(m.last_step, **res)
            reporter.report()
            pbar.update(1)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Training', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)

