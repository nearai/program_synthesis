import os
import tqdm

from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.basic_pipes import (JsonLoader, RandomAccessFile, Batch, DropKeys, KeepKeys,
                                                      WeightedMerge, EndlessShuffleCycle, Identity, SortBatchByLen,
                                                      LimitOutput)
from program_synthesis.naps.pipes.uast_pipes import SelectPseudocode, SkipPartial, ShuffleVariables


BASE_PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_PATH, "../../../data/naps/")
VERSION = "1.0"
TRAIN_A_PATH = os.path.join(DATA_FOLDER, "naps.trainA.{version}.jsonl".format(version=VERSION))
TRAIN_B_PATH = os.path.join(DATA_FOLDER, "naps.trainB.{version}.jsonl".format(version=VERSION))
TEST_PATH = os.path.join(DATA_FOLDER, "naps.test.{version}.jsonl".format(version=VERSION))


def read_naps_dataset():
    trainA = Compose([
        open(TRAIN_A_PATH),
        JsonLoader(),
        SelectPseudocode(text_key="text", texts_key="texts"),
        DropKeys(["texts", "is_training"])
    ])

    trainB = Compose([
        open(TRAIN_B_PATH),
        JsonLoader()
    ])

    test = Compose([
        open(TEST_PATH),
        JsonLoader()
    ])
    return trainA, trainB, test


def read_naps_dataset_batched(batch_size=100, trainB_weight=0.3, max_num_steps=None, shuffle_variables=False,
                              sort_batch=False):
    trainA = Compose([
        RandomAccessFile(TRAIN_A_PATH),
        JsonLoader(),
        EndlessShuffleCycle(),
        SelectPseudocode(text_key="text", texts_key="texts")
    ])

    trainB = Compose([
        RandomAccessFile(TRAIN_B_PATH),
        JsonLoader(),
        EndlessShuffleCycle(),
        SkipPartial(is_partial_key="is_partial")
    ])

    train = Compose([
        WeightedMerge(input_pipes=[trainA, trainB], p=[1.0, trainB_weight]),
        ShuffleVariables(code_tree_key="code_tree", code_sequence_key="code_sequence", text_key="text")
        if shuffle_variables else Identity(),
        KeepKeys(["text", "code_sequence"]),
        Batch(batch_size=batch_size),
        LimitOutput(max_output_num=max_num_steps) if max_num_steps else Identity(),
        SortBatchByLen(key="text") if sort_batch else Identity()
    ])

    test = Compose([
        RandomAccessFile(TEST_PATH),
        JsonLoader(),
        SkipPartial(is_partial_key="is_partial"),
        Batch(batch_size=batch_size),
        SortBatchByLen(key="text") if sort_batch else Identity()
    ])
    return train, test


# Example of iterating over the batched pipeline.
if __name__ == "__main__":
    train_dataset, test_dataset = read_naps_dataset_batched(max_num_steps=100, shuffle_variables=True, sort_batch=True)
    with train_dataset, test_dataset:
        with tqdm.tqdm() as pbar:
            for batch in train_dataset:
                pbar.update(1)
            for batch in test_dataset:
                pbar.update(1)
