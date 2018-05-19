import itertools
import torch

from program_synthesis import datasets
from program_synthesis.datasets import data
from program_synthesis.datasets.karel import refine_env
from program_synthesis.datasets.karel.refine_env import MutationActionSpace
from program_synthesis.models import karel_model, prepare_spec
from .config import MAX_TOKEN_PER_CODE


class KarelEditEnv(object):
    def __init__(self):
        self.dataset = datasets.dataset.KarelTorchDataset(
            datasets.dataset.relpath('../../data/karel/{}{}.pkl'.format('train', '')), lambda x: x)
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(datasets.dataset.relpath('../../data/karel/word.vocab')), 0)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset,
            1, collate_fn=lambda x: x, num_workers=0, pin_memory=False, shuffle=True
        )
        self.data_iter = self.dataset_loader.__iter__()
        self._cur_env = None

    def reset(self):
        """
        :return: (Task, Observation/State)
        """
        sample = self.data_iter.next()  # returns list of 1 element.
        self._cur_env = refine_env.KarelRefineEnv(sample[0].input_tests)
        obs = self._cur_env.reset()
        input_grids, output_grids = karel_model.encode_io_grids(sample)
        return (input_grids, output_grids), self.prepare_obs(obs)

    @property
    def atree(self):
        return self._cur_env.atree

    @staticmethod
    def prepare_tasks(tasks):
        input_grids = torch.cat([t[0] for t in tasks], dim=0)
        output_grids = torch.cat([t[1] for t in tasks], dim=0)
        return input_grids, output_grids

    @staticmethod
    def prepare_states(states):
        padded_states = torch.zeros(len(states), MAX_TOKEN_PER_CODE)
        for idx, state in enumerate(states):
            padded_states[idx][:state.shape[1]] = state[0]
        return padded_states

    def prepare_obs(self, obs):
        current_code = prepare_spec.lists_padding_to_tensor(
            [obs['code']], self.vocab.stoi, cuda=False, volatile=True
        )
        # current_code = prepare_spec.lists_to_packed_sequence(
        #     [obs['code']], self.vocab.stoi, cuda=False, volatile=True)
        return current_code

    def recover_code(self, state):
        return ' '.join(
                        itertools.takewhile(
                            lambda token: token != '</S>',
                            map(self.vocab.itos, state[1:])
                        )
        )

    @staticmethod
    def prepare_actions(actions):
        return actions

    def step(self, action):
        """ Execute one action on the environment

        :param action:
        :return: Observation/State, Reward/int, Done/bool, info/dic
        """
        obs, reward, done, info = self._cur_env.step(action)
        return self.prepare_obs(obs), reward, done, info
