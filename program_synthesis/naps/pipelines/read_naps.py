import os
import random


from text2code.pipes.compose import Compose
from text2code.pipes.basic_pipes import JsonLoader, Cache, Cycle, WeightedMerge, Batch, DropKeys


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


def read_naps_dataset_cached_batched(batch_size=100, num_epochs=3, trainB_weight=10):
    trainA = Compose([
        open(TRAIN_A_PATH),
        JsonLoader(),
        Cache(),
        Cycle(shuffle=True, times=num_epochs),
        SelectPseudocode,
        DropKeys(["texts", "is_training"])
    ])

    trainB = Compose([
        open(TRAIN_B_PATH),
        JsonLoader(),
        Cache(),
        Cycle(shuffle=True, times=num_epochs*trainB_weight)
    ])

    train = Compose([
        WeightedMerge(input_pipes=[trainA, trainB], weights=[1, 1]),
        Batch(batch_size=batch_size)
    ])

    test = Compose([
        open(TEST_PATH),
        JsonLoader(),
        Batch(batch_size=batch_size)
    ])
    return train, test


if __name__ == "__main__":
    train_dataset, test_dataset = read_naps_dataset_cached_batched()
    with train_dataset, test_dataset:
        for batch in train_dataset:
            pass
        for batch in test_dataset:
            pass
