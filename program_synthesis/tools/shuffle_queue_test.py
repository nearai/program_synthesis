import itertools

import pytest

from program_synthesis.datasets import shuffle_queue


def test_empty():
    queue = shuffle_queue.ShuffleQueue(buffer_size=10, iterator=[])
    with pytest.raises(StopIteration):
        next(iter(queue))


def test_small_buffer_infinite():
    queue = shuffle_queue.ShuffleQueue(
        buffer_size=5, iterator=itertools.count(0))

    for i in range(10):
        assert next(queue) < i + 5


def test_small_buffer_finite():
    queue = shuffle_queue.ShuffleQueue(buffer_size=5, iterator=range(10))
    extracted = list(queue)

    assert sorted(extracted) == range(10)
