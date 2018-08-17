import torch

from program_synthesis.karel import dataset
from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.rl_agent import utils
from program_synthesis.karel.rl_agent.utils import State


class KarelEditEnv(object):
    def __init__(self, max_token_per_code=None, mode='train'):
        self.dataset = dataset.dataset.KarelTorchDataset(dataset.dataset.relpath(f"../../data/karel/{mode}.pkl"),
                                                         lambda x: x)

        self.dataset_loader = torch.utils.data.DataLoader(self.dataset,
                                                          1, collate_fn=lambda x: x, num_workers=0, pin_memory=False,
                                                          shuffle=True)

        self.vocab = data.PlaceholderVocab(data.load_vocab(dataset.dataset.relpath('../../data/karel/word.vocab')), 0)

        self.data_iter = self.dataset_loader.__iter__()

        self.task = None
        self._env = None
        self._sample = None
        self._max_token_per_code = max_token_per_code

    def reset(self) -> (utils.Task, torch.Tensor):
        (sample,) = self.data_iter.next()  # returns list of 1 element.
        self._sample = sample  # Allow access to current sample

        self._env = refine_env.KarelRefineEnv(sample.input_tests, self._max_token_per_code)
        _ = self._env.reset()

        self.task = utils.prepare_task(sample.input_tests)

        return State(self.task, self._env.code)

    @property
    def atree(self):
        return self._env.atree

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def sample(self):
        return self._sample

    def step(self, action):
        """ Execute one action on the environment

        :param action:
        :return: Observation/State, Reward/int, Done/bool, info/dic
        """
        obs, reward, done, info = self._env.step(action)
        code = obs['code']
        return State(self.task, code), reward, done, info
