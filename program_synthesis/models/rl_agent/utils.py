import collections
import io

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, erase_factor):
        self.max_size = max_size
        self.erase_factor = erase_factor
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.extend(experience)
        if len(self.buffer) >= self.max_size:
            self.buffer = self.buffer[int(self.erase_factor * self.size):]

    def sample(self, size):
        replace_mode = size > len(self.buffer)
        index = np.random.choice(self.size, size=size, replace=replace_mode)
        return [self.buffer[idx] for idx in index]


class StepExample(collections.namedtuple('StepExample', ['task', 'state', 'action', 'reward', 'new_state'])):
    def __str__(self):
        buff = io.StringIO()

        print("State:", self.state, file=buff)
        print(self.state.shape, file=buff)
        print("Action: {} Reward: {}".format(self.action, self.reward), file=buff)

        return buff.getvalue()
