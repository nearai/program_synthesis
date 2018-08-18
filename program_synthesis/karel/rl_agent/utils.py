import collections

import numpy as np
import torch

Task = collections.namedtuple('Task', ('inputs', 'outputs'))

State = collections.namedtuple('State', ('task', 'code'))


class StepExample(collections.namedtuple('StepExample', ['state', 'action', 'reward', 'done', 'next_state'])):
    pass


class ReplayBuffer(object):
    """ Cyclic buffer
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.pointer = 0

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        if isinstance(experience, list):
            raise ValueError("Experience must be added one by one")

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.pointer += 1
        else:
            if self.pointer == self.max_size:
                self.pointer = 0

            self.buffer[self.pointer] = experience
            self.pointer += 1

    def sample(self, size):
        replace_mode = size > len(self.buffer)
        index = np.random.choice(self.size, size=size, replace=replace_mode)
        return [self.buffer[idx] for idx in index]


def prepare_code(code, vocab, tensor=False):
    code_seq = [vocab.stoi(token) for token in code]

    if tensor:
        return torch.LongTensor(code_seq).unsqueeze(0)
    else:
        return code_seq


def prepare_task(tests) -> Task:
    input_grids = []
    output_grids = []

    def to_grid(ix):
        grid = np.zeros((1, 15, 18, 18))
        grid.ravel()[ix] = 1
        return torch.from_numpy(grid).to(torch.float32)

    for test in tests:
        I, O = map(to_grid, [test['input'], test['output']])
        input_grids.append(I)
        output_grids.append(O)

    input_grids = torch.cat(input_grids, dim=0).unsqueeze(0)
    output_grids = torch.cat(output_grids, dim=0).unsqueeze(0)
    return Task(input_grids, output_grids)
