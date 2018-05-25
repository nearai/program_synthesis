import numpy as np


class ShuffleQueue(object):
    '''Obtains items from `iterator` and returns them in a random order.'''
    def __init__(self, buffer_size, iterator, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.buffer = []
        self.iterator = iter(iterator)
        self.exhausted = False

        # Fill buffer
        try:
            while len(self.buffer) < buffer_size:
                self.buffer.append(next(self.iterator))
        except StopIteration:
            self.exhausted = True

    def __iter__(self):
        return self

    def next(self):
        if not self.buffer:
            raise StopIteration

        if self.exhausted:
            return self.buffer.pop()

        index = self.rng.choice(len(self.buffer))
        to_return = self.buffer[index]
        try:
            self.buffer[index] = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            del self.buffer[index]
            self.rng.shuffle(self.buffer)
        return to_return
    
    __next__ = next
