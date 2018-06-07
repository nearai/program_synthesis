import os
import random
import tqdm

from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.basic_pipes import JsonLoader, RandomAccessFile, Cycle, Merge, Batch, DropKeys


SelectPseudocode = lambda d: {**d, **{"text": random.choice(d["texts"])}}

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
        SelectPseudocode,
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


def read_naps_dataset_batched(batch_size=100, num_epochs=300, trainB_weight=10):
    trainA = Compose([
        RandomAccessFile(TRAIN_A_PATH),
        JsonLoader(),
        Cycle(shuffle=True, times=num_epochs),
        SelectPseudocode,
        DropKeys(["texts", "is_training"])
    ])

    trainB = Compose([
        RandomAccessFile(TRAIN_B_PATH),
        JsonLoader(),
        Cycle(shuffle=True, times=num_epochs*trainB_weight)
    ])

    train = Compose([
        Merge(input_pipes=[trainA, trainB], mode='random'),
        Batch(batch_size=batch_size)
    ])

    test = Compose([
        RandomAccessFile(TEST_PATH),
        JsonLoader(),
        Batch(batch_size=batch_size)
    ])
    return train, test


# Example of iterating over the batched pipeline.
if __name__ == "__main__":
    train_dataset, test_dataset = read_naps_dataset_batched()
    with train_dataset, test_dataset:
        with tqdm.tqdm(total=len(train_dataset)+len(test_dataset)) as pbar:
            for batch in train_dataset:
                pbar.update(1)
            for batch in test_dataset:
                pbar.update(1)
