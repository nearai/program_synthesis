""" Karel Agent - Reinforcement Learning

    More information on program_synthesis/models/rl_agent/description.md

"""

import numpy as np
import torch
import torch.utils.data
from program_synthesis.datasets.karel import mutation
from torch import optim

from program_synthesis.datasets.karel.refine_env import MutationActionSpace
from program_synthesis.models.rl_agent.config import VOCAB_SIZE, EPSILON, DISCOUNT, ALPHA, TOTAL_MUTATION_ACTIONS, \
    MAX_TOKEN_PER_CODE, LOCATION_EMBED_SIZE
from program_synthesis.models.rl_agent.environment import KarelEditEnv
from program_synthesis.models.rl_agent.policy import KarelEditPolicy
from program_synthesis.models.rl_agent.utils import ReplayBuffer, StepExample, Action
from program_synthesis.tools import saver


# https://discuss.pytorch.org/t/nn-criterions-dont-compute-the-gradient-w-r-t-targets/3693
def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


# https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960/2
def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y.type(torch.long)]


class KarelAgent:
    def __init__(self, env, args):
        self.vocab = env.vocab
        self.env = env
        self.task_enc = None

        # Build model
        self.model = KarelEditPolicy(VOCAB_SIZE, args)
        self.criterion = mse_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def set_task(self, task):
        self.task_enc = self.model.encode_task(task)

    @staticmethod
    def policy_from_value(action_value, mode='linear'):
        if mode == "linear":
            act_dist = action_value / action_value.sum()
        elif mode == "softmax":
            act_dist = torch.exp(action_value)
            act_dist /= act_dist.sum()
        else:
            raise ValueError('Mode {mode} is invalid'.format(mode=mode))

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
                    return Action(action_id, params)
        else:
            action_value = self.model.action_value(self.task_enc, code)
            act_dist = self.policy_from_value(action_value, mode='softmax')

            act_dist = act_dist * mask
            act_dist /= act_dist.sum()

            while True:
                action_id = np.random.choice(np.arange(8), p=act_dist.numpy().ravel())

                # Select parameters randomly (TODO: Select parameter according to learned parameter values network)
                params = action_space.sample_parameters(action_id)

                if params is not None:
                    break

            return Action(action_id, params)

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
        batch_size = targets.shape[0]

        # Compute action gradients
        self.optimizer.zero_grad()
        tasks_encoded = self.model.encode_task(tasks)
        action_values = self.model.action_value(tasks_encoded, states)
        current_value = torch.Tensor([action_values[idx][actions[idx][0]] for idx in range(batch_size)])
        loss = self.criterion(current_value, targets)

        # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2?
        # Is it better to compute multiple loss once?
        loss.backward(retain_graph=True)

        self.optimizer.step()

        # Update location network
        self.optimizer.zero_grad()

        masks = torch.ones(batch_size, MAX_TOKEN_PER_CODE)
        selected_location = torch.zeros(batch_size, MAX_TOKEN_PER_CODE)

        for idx in range(batch_size):
            if actions[idx].id in (mutation.ADD_ACTION, mutation.REPLACE_ACTION, mutation.REMOVE_ACTION):
                selected_location[idx][actions[idx].parameters.location] = 1.

        action_categorical = to_categorical(torch.Tensor([a.id for a in actions]), TOTAL_MUTATION_ACTIONS)
        states_encoded = to_categorical(states, VOCAB_SIZE)

        location_reward_vec, location_embed_vec = self.model.location(action_categorical, tasks_encoded, states_encoded,
                                                                      masks)
        location_reward_vec = location_reward_vec.reshape(batch_size, -1)

        has_location = torch.ones(batch_size)

        location_reward = (location_reward_vec * selected_location).sum(1)
        location_reward_target = targets * has_location

        location_loss = self.criterion(location_reward, location_reward_target)

        location_loss.backward(retain_graph=True)
        self.optimizer.step()

        # Update token network
        self.optimizer.zero_grad()

        location_embed = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        token_selected = torch.zeros(batch_size, len(mutation.ACTION_NAMES))

        for idx in range(batch_size):
            if actions[idx].id in (mutation.ADD_ACTION, mutation.REPLACE_ACTION):
                location_embed[idx] = location_embed_vec[idx][actions[idx].parameters.location]
                token_selected[idx][mutation.get_action_name_id(actions[idx].parameters.token)] = 1.

        token_value = self.model.karel_token(action_categorical, tasks_encoded, location_embed)
        token_reward = (token_selected * token_value).sum(1)

        token_loss = mse_loss(token_reward, targets)
        token_loss.backward()

        self.optimizer.step()

        print("train step...")

    def update(self, other):
        self.model.load_state_dict(other.model.state_dict())
