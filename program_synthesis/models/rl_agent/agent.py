""" Karel Agent - Reinforcement Learning

    More information on program_synthesis/models/rl_agent/description.md

"""

import numpy as np
import torch
import torch.utils.data
from torch import optim

from program_synthesis.datasets.karel import mutation
from program_synthesis.datasets.karel.mutation import ACTION_NAMES, BLOCK_TYPE, Action, ActionAddParameters, \
    ActionRemoveParameters, ActionReplaceParameters, ActionUnwrapBlockParameters, ActionWrapBlockParameters, \
    ActionWrapIfElseParameters
from program_synthesis.datasets.karel.refine_env import MutationActionSpace
from program_synthesis.models.rl_agent.config import VOCAB_SIZE, TOTAL_MUTATION_ACTIONS, \
    MAX_TOKEN_PER_CODE, LOCATION_EMBED_SIZE, TOKEN_EMBED_SIZE
from program_synthesis.models.rl_agent.policy import KarelEditPolicy


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
    def policy_from_value(action_value, mode='softmax'):
        if mode == "linear":
            act_dist = action_value / action_value.sum()
        elif mode == "softmax":
            if isinstance(action_value, np.ndarray):
                act_dist = np.exp(action_value)
            else:
                act_dist = torch.exp(action_value)
            act_dist /= act_dist.sum()
        else:
            raise ValueError('Mode {mode} is invalid'.format(mode=mode))

        return act_dist

    @staticmethod
    def choice(reward, mask=None, mode='softmax'):
        """
        :param reward: np.ndarray
        :param mask: np.ndarray
        :param mode: ('linear', 'softmax')
        """
        if not isinstance(reward, np.ndarray):
            reward = reward.numpy()
        assert sum([1 for x in reward.shape if x > 1]) <= 1

        reward = reward.reshape(-1)
        dist = KarelAgent.policy_from_value(reward, mode=mode)

        if mask is not None:
            mask = mask.reshape(-1)
            dist *= mask
            dist /= dist.sum()

        if dist.sum() <= 1e-12:
            print("Warning")
            return None

        return np.random.choice(range(dist.shape[0]), p=dist)

    def select_action(self, code, epsilon_greedy):
        """ Select action (with parameters)

            Note:
                Call `set_task` first
                The code is linked to the agent via the environment
        """
        with torch.no_grad():
            action_space = MutationActionSpace(atree=self.env.atree)

            # TODO[IMPLEMENT_ACTIONS]: This mask is to select only implemented actions
            implemented_actions = torch.Tensor([1., 1., 1., 1., 1., 1., 0., 0.])

            if epsilon_greedy is not None and np.random.random() < epsilon_greedy:
                while True:
                    # Select action randomly
                    action_id = self.choice(implemented_actions.numpy().ravel(), mode='linear')

                    # Select parameters randomly
                    params = action_space.sample_parameters(action_id)

                    if params is not None:
                        return Action(action_id, params)
            else:
                action_value = self.model.action_value(self.task_enc, code)
                act_dist = self.policy_from_value(action_value)

                act_dist = act_dist * implemented_actions
                act_dist /= act_dist.sum()

                # <DEBUG ACTION_TYPE>
                # act_dist = torch.zeros(8)
                # act_dist[mutation.ADD_ACTION] = 1.
                # act_dist[mutation.REMOVE_ACTION] = 1.
                # act_dist[mutation.REPLACE_ACTION] = 1.
                # act_dist[mutation.UNWRAP_BLOCK] = 1.
                # act_dist[mutation.WRAP_BLOCK] = 1.
                # act_dist /= act_dist.sum()
                # <END DEBUG>

                params = None
                action_id = None

                while params is None:
                    action_id = np.random.choice(np.arange(8), p=act_dist.numpy().ravel())

                    print("ACTION ID:", action_id)

                    if action_id == mutation.ADD_ACTION:
                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        # Select location
                        location_mask = torch.zeros(code.shape[-1], 1)
                        location_mask.reshape(-1)[list(self.env.atree.add_action_locs)] = 1.

                        location_embedding = self.model.location(action_categorical, self.model.state_enc,
                                                                 self.model.code_token_embed)

                        location_reward = self.model.single_location(location_embedding)
                        selected_location = self.choice(location_reward.numpy().ravel(), location_mask.numpy().ravel())

                        selected_location_embed = location_embedding[0][selected_location].view(1, -1)

                        # Select karel token
                        karel_token_reward = self.model.karel_token(action_categorical, self.model.state_enc,
                                                                    selected_location_embed)

                        selected_action_type = self.choice(karel_token_reward.numpy().reshape(-1))

                        params = ActionAddParameters(selected_location, ACTION_NAMES[selected_action_type])

                    elif action_id == mutation.REMOVE_ACTION:
                        # If there is only one token it can't be removed

                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        if len(self.env.atree.remove_action_locs) == 0:
                            continue

                        # Select location
                        location_mask = torch.zeros(code.shape[-1], 1)
                        location_mask.reshape(-1)[list(self.env.atree.remove_action_locs)] = 1.

                        location_embedding = self.model.location(action_categorical, self.model.state_enc,
                                                                 self.model.code_token_embed)

                        location_reward = self.model.single_location(location_embedding)

                        selected_location = self.choice(location_reward, location_mask)

                        params = ActionRemoveParameters(selected_location)

                    elif action_id == mutation.REPLACE_ACTION:
                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        if len(self.env.atree.replace_action_locs) == 0:
                            continue

                        # Select location
                        location_mask = torch.zeros(code.shape[-1], 1)
                        location_mask.reshape(-1)[list(self.env.atree.replace_action_locs)] = 1.

                        location_embedding = self.model.location(action_categorical, self.model.state_enc,
                                                                 self.model.code_token_embed)

                        location_reward = self.model.single_location(location_embedding)

                        selected_location = self.choice(location_reward, location_mask)
                        selected_location_embed = location_embedding[0][selected_location].view(1, -1)

                        # Select karel token
                        karel_token_reward = self.model.karel_token(action_categorical, self.model.state_enc,
                                                                    selected_location_embed)

                        selected_action_type = self.choice(karel_token_reward)

                        params = ActionReplaceParameters(selected_location, ACTION_NAMES[selected_action_type])

                    elif action_id == mutation.UNWRAP_BLOCK:

                        # If there is only one token it can't be removed

                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        if len(self.env.atree.unwrap_block_locs) == 0:
                            continue

                        # Select location
                        location_mask = np.zeros(code.shape[-1])
                        location_mask[list(self.env.atree.unwrap_block_locs)] = 1.

                        location_embedding = self.model.location(action_categorical,
                                                                 self.model.state_enc,
                                                                 self.model.code_token_embed)

                        location_reward = self.model.single_location(location_embedding)
                        selected_location = self.choice(location_reward, location_mask)

                        params = ActionUnwrapBlockParameters(selected_location)

                    elif action_id == mutation.WRAP_BLOCK:
                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        block_type_reward = self.model.block_type(self.model.state_enc)
                        selected_block_type = self.choice(block_type_reward.numpy().reshape(-1))

                        if selected_block_type in (0, 1):  # (`if`, `while`)
                            block_type_categorical = torch.eye(3)[selected_block_type].reshape(1, -1)
                            condition_id_reward = self.model.condition_id(self.model.state_enc, block_type_categorical)
                            condition_id = self.choice(condition_id_reward)
                        else:
                            condition_id_reward = self.model.repeat_count(self.model.state_enc)
                            condition_id = self.choice(condition_id_reward)

                        location_embedding = self.model.location(action_categorical, self.model.state_enc,
                                                                 self.model.code_token_embed)

                        loc_pair = []
                        loc0_embed, loc1_embed = [], []

                        for loc0, loc1 in action_space.enumerate_simple_wrap_spans():
                            loc_pair.append((loc0, loc1))
                            loc0_embed.append(location_embedding[0, loc0:loc0 + 1])
                            loc1_embed.append(location_embedding[0, loc1:loc1 + 1])

                        if len(loc_pair) == 0:
                            continue

                        loc_pair_reward = self.model.double_location(torch.cat(loc0_embed), torch.cat(loc1_embed))
                        selected_loc_pair = self.choice(loc_pair_reward)

                        start, end = loc_pair[selected_loc_pair]

                        params = ActionWrapBlockParameters(BLOCK_TYPE[selected_block_type], condition_id, start, end)

                    elif action_id == mutation.WRAP_IFELSE:
                        action_categorical = torch.zeros(1, 8)
                        action_categorical.reshape(-1)[action_id] = 1.

                        block_type_categorical = torch.eye(3)[0].reshape(1, -1)
                        condition_id_reward = self.model.condition_id(self.model.state_enc, block_type_categorical)
                        condition_id = self.choice(condition_id_reward)

                        location_embedding = self.model.location(action_categorical, self.model.state_enc,
                                                                 self.model.code_token_embed)

                        loc_tuple = []
                        loc0_embed, loc1_embed, loc2_embed = [], [], []

                        for loc0, loc1, loc2 in action_space.enumerate_composite_wrap_spans():
                            loc_tuple.append((loc0, loc1, loc2))
                            loc0_embed.append(location_embedding[0, loc0:loc0 + 1])
                            loc1_embed.append(location_embedding[0, loc1:loc1 + 1])
                            loc2_embed.append(location_embedding[0, loc2:loc2 + 1])

                        if len(loc_tuple) == 0:
                            continue

                        loc_pair_reward = self.model.triple_location(torch.cat(loc0_embed),
                                                                     torch.cat(loc1_embed),
                                                                     torch.cat(loc2_embed))
                        selected_loc_pair = self.choice(loc_pair_reward)

                        if_start, else_start, end = loc_tuple[selected_loc_pair]

                        params = ActionWrapIfElseParameters(condition_id, if_start, else_start, end)

                    elif action_id == mutation.REPLACE_COND:
                        raise NotImplementedError()

                    elif action_id == mutation.SWITCH_IF_WHILE:
                        raise NotImplementedError()

                    else:
                        raise ValueError(f"Invalid action id {action}. \
                        Action id must be in the range [0, {TOTAL_MUTATION_ACTIONS})")

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
        # TODO: Fix loss

        batch_size = targets.shape[0]

        # Compute action gradients
        self.optimizer.zero_grad()
        tasks_encoded = self.model.encode_task(tasks)
        action_values = self.model.action_value(tasks_encoded, states)
        current_value = torch.Tensor([action_values[idx][actions[idx][0]] for idx in range(batch_size)])
        loss = self.criterion(current_value, targets)

        # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2
        # Is it better to compute multiple loss once?
        loss.backward(retain_graph=True)

        self.optimizer.step()

        # Update location network
        self.optimizer.zero_grad()

        selected_location = torch.zeros(batch_size, MAX_TOKEN_PER_CODE)

        for idx in range(batch_size):
            if actions[idx].id in (mutation.ADD_ACTION, mutation.REPLACE_ACTION, mutation.REMOVE_ACTION):
                selected_location[idx][actions[idx].parameters.location] = 1.

        action_categorical = to_categorical(torch.Tensor([a.id for a in actions]), TOTAL_MUTATION_ACTIONS)
        states_encoded = self.model.code_encoder.get_embed(states).view(MAX_TOKEN_PER_CODE, batch_size,
                                                                        TOKEN_EMBED_SIZE)

        location_reward_vec, location_embed_vec = self.model.location(action_categorical, tasks_encoded, states_encoded)
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
