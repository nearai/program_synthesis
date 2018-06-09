import collections
import copy

import numpy as np

import torch
import torch.nn as nn
from torch import optim

from program_synthesis.common.tools import saver

from program_synthesis.karel.dataset import dataset
from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.dataset import mutation

from program_synthesis.karel.models import karel_model
from program_synthesis.karel.models import prepare_spec
from program_synthesis.karel.models.modules import karel
from program_synthesis.karel.models.modules import karel_common
from program_synthesis.karel.models.modules import karel_edit


DISCOUNT = 0.9
EPISLON = 0.1
ALPHA = 0.7


#  https://arxiv.org/pdf/1511.04143.pdf

class KarelEditEnv(object):

    def __init__(self):
        self.dataset = dataset.KarelTorchDataset(
            dataset.dataset.relpath('../../data/karel/{}{}.pkl'.format('train', '')), lambda x: x)
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(dataset.dataset.relpath('../../data/karel/word.vocab')), 0)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset,
            1, collate_fn=lambda x: x, num_workers=0, pin_memory=False, shuffle=True
        )
        self.data_iter = self.dataset_loader.__iter__()
        self._cur_env = None

    def reset(self):
        example = self.data_iter.next()  # returns list of 1 element.
        self._cur_env = refine_env.KarelRefineEnv(example[0].input_tests)
        obs = self._cur_env.reset()
        input_grids, output_grids = karel_model.encode_io_grids(example)
        return (input_grids, output_grids), self.prepare_obs(obs)

    def prepare_tasks(self, tasks):
        input_grids = torch.cat([t[0] for t in tasks], dim=0)
        output_grids = torch.cat([t[1] for t in tasks], dim=0)
        return (input_grids, output_grids)

    def prepare_states(self, states):
        return torch.cat(states, dim=0)

    def prepare_obs(self, obs):
        current_code = prepare_spec.lists_padding_to_tensor(
            [obs['code']], self.vocab.stoi, cuda=False, volatile=True
        )
        # current_code = prepare_spec.lists_to_packed_sequence(
        #     [obs['code']], self.vocab.stoi, cuda=False, volatile=True)
        return current_code

    def prepare_actions(self, actions):
        return actions

    def step(self, action):
        obs, reward, done, info = self._cur_env.step(action)
        return self.prepare_obs(obs), reward, done, info


class KarelEditPolicy(nn.Module):
    def __init__(self, vocab_size, args):
        super(KarelEditPolicy, self).__init__()
        self.args = args

        self.io_encoder = karel_common.make_task_encoder(args)
        self.code_encoder = karel.CodeEncoder(vocab_size, args)

        # Action decoder.
        self.action_type = nn.Linear(100, 5)
        self.action_pointer = karel_edit.ScaledDotProductPointer(100, 100)
        self.action_token = nn.Linear(100, vocab_size, bias=False)

    def encode(self, input_grid, output_grid):
        return self.io_encoder(input_grid, output_grid)

    def encode_code(self, code_state):
        return self.code_encoder(code_state)

    def action_value(self, task_state, code_state):
        current_code_enc = self.encode_code(code_state)
        state = torch.cat([task_state, current_code_enc.state])
        act = self.action_type(state)
        ptr = self.action_pointer(state)
        token = self.action_token(state)
        return act, ptr, token


class KarelAgent(object):

    def __init__(self, env, args):
        self.vocab = env.vocab
        self.model = KarelEditPolicy(30, args)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.task_state = None

    def set_task(self, task):
        input_grids, output_grids = task
        self.task_state = self.model.encode(input_grids, output_grids)

    def select_action(self, state, epsilon_greedy):
        return (mutation.ADD_ACTION, (3, 'move'))
        assert self.task_state is not None
        if epsilon_greedy is not None and np.random.random() < epsilon_greedy:
            return np.random.randint(state.size)
        _, action = self.best_action_value([state])
        return action[0]

    def action_value(self, tasks, states):
        input_grids, output_grids = tasks
        task_state = self.model.encode(input_grids, output_grids)
        return self.model.action_value(task_state, states)

    def best_action_value(self, tasks, states):
        input_grids, output_grids = tasks
        task_state = self.model.encode(input_grids, output_grids)
        (action_edit_values, action_position_values, action_token_values) = self.model.action_value(task_state, states)
        edit_prob, edit_value = action_edit_values[0].max(1)
        position_prob, position_value = action_position_values[0].max(1)
        token_prob, token_value = action_token_values.values[0].max(1)
        return (edit_prob, position_prob, token_prob), (edit_value, position_value, token_value)

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


StepExample = collections.namedtuple('StepExample', ['task', 'state', 'action', 'reward', 'new_state'])


def rollout(env, agent, epsilon_greedy, max_rollout_length):
    eps = EPISLON if epsilon_greedy else None
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
        tasks = self.env.prepare_tasks([ex.task for ex in batch])
        states = self.env.prepare_states([ex.state for ex in batch])
        new_states = self.env.prepare_states([ex.new_state for ex in batch])
        actions = self.env.prepare_actions([ex.action for ex in batch])
        targets = np.zeros(len(batch))
        # prepare_batch(batch)
        value = self.critic.action_value(states)
        new_value, _ = self.actor.best_action_value(new_states)
        for idx, ex in enumerate(batch):
            Q_s_a = value[idx][ex.action]
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
        karel_io_enc='lgrl', lr=0.001)

    agent_cls = KarelAgent
    env = KarelEditEnv()
    # experience = rollout(env, agent_cls(env, args), False, 1)
    # for x in experience:
    #     print(x)
    # env.reset()
    # print(env.step((mutation.ADD_ACTION, (3, 'move'))))
    trainer = PolicyTrainer(args, agent_cls, env)
    trainer.train()


if __name__ == "__main__":
    main()
