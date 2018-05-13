""" Karel Agent - Reinforcement Learning

    More information on program_synthesis/models/karel-rl.md

    Ideas:
        + Hindsight Experience Replay
        + Prioritized Experience Replay
        + Meta learning (MAML, Reptile)
        + Hierarchical methods

    Reference:
        Deep RL in parametrized action space
        https://arxiv.org/pdf/1511.04143.pdf
"""
import collections
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim

from program_synthesis import datasets
from program_synthesis.datasets import data
from program_synthesis.datasets.karel import refine_env, mutation
from program_synthesis.datasets.karel.refine_env import MutationActionSpace
from program_synthesis.models import karel_model
from program_synthesis.models import prepare_spec
from program_synthesis.models.karel_agent_model import ActionTypeModel, LocationParameterModel, KarelTokenParameterModel
from program_synthesis.models.modules import karel
from program_synthesis.models.modules import karel_common
from program_synthesis.tools import saver

DISCOUNT = 0.99
EPSILON = 0.1
ALPHA = 0.7

VOCAB_SIZE = 30
MAX_TOKEN_PER_CODE = 20
TOKEN_EMBED_SIZE = 256


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


class KarelEditPolicy(nn.Module):
    def __init__(self, vocab_size, args):
        super(KarelEditPolicy, self).__init__()
        self.args = args

        self.io_encoder = karel_common.make_task_encoder(args)
        self.code_encoder = karel.CodeEncoderAlternative(vocab_size, args)

        self.action_type = ActionTypeModel()

        # Model to determine parameters
        self.location = LocationParameterModel()
        self.karel_token = KarelTokenParameterModel()

        self._code_embed = None
        self._state = None

    def encode_task(self, tasks):
        input_grids, output_grids = tasks
        io_pair_enc = self.io_encoder(input_grids, output_grids)
        max_value, _ = io_pair_enc.max(1)  # TODO: What about this final encoding
        return max_value

    def encode_code(self, code):
        """
            :return: (code_embedded, code_latent_representation)
        """
        return self.code_encoder(code)

    def action_value(self, task_enc, code):
        """ Compute expected reward of executing every action (without parameters)

            Note:
                Store the embedded code in the policy for picking parameters
        """
        self._code_embed, code_enc = self.encode_code(code)
        self._state = torch.cat([task_enc, code_enc], 1)
        act = self.action_type(self._state)
        return act

    def parameters_value(self, action):
        """
            Warning:
                This method must be called after calling `action_value`
        """
        if action == mutation.ADD_ACTION:
            pass
        elif action == mutation.REMOVE_ACTION:
            pass
        elif action == mutation.REPLACE_ACTION:
            pass
        else:
            raise NotImplementedError("Action parameters values not implemented yet.")

    def forward(self):
        pass


class KarelAgent:
    def __init__(self, env, args):
        self.vocab = env.vocab
        self.env = env
        self.task_enc = None

        # Build model
        self.model = KarelEditPolicy(VOCAB_SIZE, args)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def set_task(self, task):
        self.task_enc = self.model.encode_task(task)

    @staticmethod
    def get_distribution_from_values(action_value, mode='linear'):
        if mode == "linear":
            act_dist = action_value / action_value.sum()
        elif mode == "softmax":
            act_dist = torch.exp(action_value)
            act_dist /= act_dist.sum()
        else:
            raise ValueError(f'Mode {mode} is invalid')

        return act_dist

    def select_action(self, code, epsilon_greedy):
        """ Select action (with parameters)

            Note:
                Call `set_task` first
        """
        action_space = MutationActionSpace(atree=self.env.atree)

        # TODO[IMPLEMENT_ACTIONS]: This mask is to select only implemented actions
        mask = torch.Tensor([1., 1., 1., 0., 0., 0., 0., 0.])

        if epsilon_greedy is not None and np.random.random() < epsilon_greedy:
            while True:
                # Select action randomly
                act_dist = mask
                act_dist /= act_dist.sum()

                action_id = np.random.choice(np.arange(8), p=act_dist.numpy().ravel())

                # Select parameters randomly
                params = action_space.sample_parameters(action_id)

                if params is not None:
                    return action_id, params
        else:
            action_value = self.model.action_value(self.task_enc, code)
            act_dist = self.get_distribution_from_values(action_value, mode='softmax')

            act_dist = act_dist * mask
            act_dist /= act_dist.sum()

            while True:
                action_id = np.random.choice(np.arange(8), p=act_dist.numpy().ravel())

                # Select parameters randomly (TODO: Select parameter according to learned parameter values network)
                params = action_space.sample_parameters(action_id)

                if params is not None:
                    return action_id, params

    def action_value(self, tasks, code):
        """ Determine the value of each action (without parameters)
        """
        tasks_encoded = self.model.encode_task(tasks)
        return self.model.action_value(tasks_encoded, code)

    def best_action_value(self, tasks, code):
        """ Determine value of best action in current state
        """
        task_state = self.model.encode_task(tasks)
        action_values = self.model.action_value(task_state, code)
        best_val, _ = action_values.max(dim=1)
        return best_val

    def train(self, tasks, states, actions, targets):
        self.optimizer.zero_grad()
        input_grids, output_grids = tasks
        task_state = self.model.encode(input_grids, output_grids)
        action_values = self.model.action_value(task_state, states)
        current_value = action_values.dot(actions)
        loss = self.criterion(current_value, targets)
        loss.backward()
        self.optimizer.step()

    def update(self, other):
        self.model.load_state_dict(other.model.state_dict())


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


# TODO: Change `state` name for `code`
class StepExample(collections.namedtuple('StepExample', ['task', 'state', 'action', 'reward', 'new_state'])):
    def __str__(self):
        buff = io.StringIO()

        print("State:", self.state, file=buff)
        print(self.state.shape, file=buff)
        print("Action: {} Reward: {}".format(self.action, self.reward), file=buff)

        return buff.getvalue()


def rollout(env, agent, epsilon_greedy, max_rollout_length):
    with torch.no_grad():
        eps = EPSILON if epsilon_greedy else None
        task, state = env.reset()
        agent.set_task(task)
        experience = []
        success = False
        for _ in range(max_rollout_length):
            action = agent.select_action(state, eps)
            new_state, reward, done, _ = env.step(action)
            experience.append(StepExample(task, state, action, reward, new_state))
            if done:
                success = True
                break
            state = new_state
        return success, experience


class PolicyTrainer(object):
    def __init__(self, args, agent_cls, env):
        self.args = args
        self.actor = agent_cls(env, args)
        self.critic = agent_cls(env, args)
        self.env = env

    def train_actor_critic(self, batch):
        # TODO: Implement train actor critic

        tasks = self.env.prepare_tasks([ex.task for ex in batch])
        states = self.env.prepare_states([ex.state for ex in batch])
        new_states = self.env.prepare_states([ex.new_state for ex in batch])
        actions = self.env.prepare_actions([ex.action for ex in batch])

        targets = np.zeros(len(batch))
        values = self.critic.action_value(tasks, states)
        new_value = self.actor.best_action_value(tasks, new_states)
        for idx, ex in enumerate(batch):
            Q_s_a = values[idx][ex.action]
            new_Q_s_a = 0 if ex.reward == 0 else (ex.reward + DISCOUNT * new_value[idx])
            targets[idx] = Q_s_a + ALPHA * (new_Q_s_a - Q_s_a)

        self.critic.train(tasks, states, actions, targets)

    def train(self):
        replay_buffer = ReplayBuffer(self.args.replay_buffer_size, self.args.erase_factor)
        for epoch in range(self.args.num_epochs):

            for _ in range(self.args.num_episodes):
                _, experience = rollout(self.env, self.actor, True, self.args.max_rollout_length)
                replay_buffer.add(experience)

            for _ in range(self.args.num_training_steps):
                batch = replay_buffer.sample(self.args.batch_size)
                self.train_actor_critic(batch)

            if (epoch + 1) % self.args.update_actor_epoch == 0:
                self.actor.update(self.critic)
                # log info here


def main():
    args = saver.ArgsDict(
        num_epochs=100, max_rollout_length=10, replay_buffer_size=16384, erase_factor=0.01,
        num_episodes=10, num_training_steps=10, batch_size=32, update_actor_epoch=10,
        karel_io_enc='lgrl', lr=0.001, cuda=False)

    agent_cls = KarelAgent
    env = KarelEditEnv()

    # success, experience = rollout(env, agent_cls(env, args), False, 10)
    #
    # for x in experience:
    #     print("*****************")
    #     print(x)
    #
    # env.reset()
    # exit(0)

    trainer = PolicyTrainer(args, agent_cls, env)
    trainer.train()


if __name__ == "__main__":
    main()
