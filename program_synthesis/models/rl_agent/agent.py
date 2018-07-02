""" Karel Agent - Reinforcement Learning

    More information on program_synthesis/models/rl_agent/description.md

"""

import time

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
    LOCATION_EMBED_SIZE, KAREL_STATIC_TOKEN, BLOCK_TYPE_SIZE, CONDITION_SIZE, REPEAT_COUNT_SIZE
from program_synthesis.models.rl_agent.logger import logger_train, logger_task
from program_synthesis.models.rl_agent.policy import KarelEditPolicy


# https://discuss.pytorch.org/t/nn-criterions-dont-compute-the-gradient-w-r-t-targets/3693
def mse_loss(input, target, mask=None):
    if mask is not None:
        input = input * mask
        target = target * mask

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
    def policy_from_value(action_value, mode='greedy'):
        if not isinstance(action_value, np.ndarray):
            action_value = action_value.numpy()

        if mode == "linear":
            act_dist = action_value / action_value.sum()
        elif mode == "softmax":
            act_dist = np.exp(action_value)
            act_dist /= act_dist.sum()
        elif mode == 'greedy':
            act_dist = np.zeros(action_value.shape)
            act_dist.ravel()[action_value.argmax()] = 1.
        else:
            raise ValueError('Mode {mode} is invalid'.format(mode=mode))

        return act_dist

    @staticmethod
    def choice(reward, mask=None, mode='greedy'):
        """
        :param reward: np.ndarray
        :param mask: np.ndarray
        :param mode: ('linear', 'softmax', 'greedy')
        """
        if not isinstance(reward, np.ndarray):
            reward = reward.numpy()

        reward = reward.reshape(-1)
        dist = KarelAgent.policy_from_value(reward, mode=mode)

        if mask is not None:
            mask = mask.reshape(-1)
            dist *= mask
            dist /= dist.sum()

        if dist.sum() <= 1e-12:
            raise ValueError()

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

                # Pick random action if current action is not available
                if act_dist.sum() < 1e-9:
                    act_dist += np.ones(act_dist.shape) * 1e-4

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

                        if len(self.env.atree.replace_action_locs) != 0:
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

                        if len(self.env.atree.unwrap_block_locs) != 0:
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

                        if len(loc_pair) != 0:
                            loc_pair_reward = self.model.double_location(torch.cat(loc0_embed), torch.cat(loc1_embed))
                            selected_loc_pair = self.choice(loc_pair_reward)

                            start, end = loc_pair[selected_loc_pair]

                            params = ActionWrapBlockParameters(BLOCK_TYPE[selected_block_type], condition_id, start,
                                                               end)

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

                        if len(loc_tuple) != 0:
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

                    if params is None:
                        # Add randomness
                        act_dist += np.ones(act_dist.shape) * 1e-1
                        act_dist /= action_value.sum()

                res_action = Action(action_id, params)
                logger_task.info(f"Selected action: {res_action}")
                return res_action

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

    def train(self, tasks, codes, actions, targets):
        # TODO: Deep train
        # Target value of each action is the maximum value among their parameters
        # Example:
        # The target value of add_action, is the maximum value among the location where a token can be added
        # The target value of the token is the real target of the example

        _begin_train = time.clock()
        batch_size = targets.shape[0]
        self.optimizer.zero_grad()

        tasks_enc = self.model.encode_task(tasks)
        action_values = self.model.action_value(tasks_enc, codes)
        code_token_embed = self.model.code_token_embed
        state_enc = self.model.state_enc

        batch_action_value = torch.zeros(batch_size)

        # noinspection PyTypeChecker
        actions_categorical = to_categorical(torch.Tensor([action.id for action in actions]), TOTAL_MUTATION_ACTIONS)
        location_embed = self.model.location(actions_categorical, state_enc, code_token_embed)
        batch_location_embed = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        batch_location_mask = torch.zeros(batch_size)

        karel_token_locations = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        karel_token_selected_mask = torch.zeros(batch_size, KAREL_STATIC_TOKEN)
        karel_token_mask = torch.zeros(batch_size)

        block_type_reward = self.model.block_type(state_enc)
        block_type_selected_mask = torch.zeros(batch_size, BLOCK_TYPE_SIZE)
        block_type_mask = torch.zeros(batch_size)

        condition_selected_mask = torch.zeros(batch_size, CONDITION_SIZE)
        condition_mask = torch.zeros(batch_size)

        repeat_count_reward = self.model.repeat_count(state_enc)
        repeat_count_selected_mask = torch.zeros(batch_size, REPEAT_COUNT_SIZE)
        repeat_count_mask = torch.zeros(batch_size)

        double_loc_0 = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        double_loc_1 = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        double_loc_mask = torch.zeros(batch_size)

        triple_loc_0 = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        triple_loc_1 = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        triple_loc_2 = torch.zeros(batch_size, LOCATION_EMBED_SIZE)
        triple_loc_mask = torch.zeros(batch_size)

        for idx, action in enumerate(actions):
            batch_action_value[idx] = action_values[idx][action.id]

            if action.id in (
                    mutation.ADD_ACTION, mutation.REMOVE_ACTION, mutation.REPLACE_ACTION, mutation.UNWRAP_BLOCK):
                batch_location_embed[idx] = location_embed[idx][action.parameters.location]
                batch_location_mask[idx] = 1.

            if action.id in (mutation.ADD_ACTION, mutation.REPLACE_ACTION):
                karel_token_locations[idx] = location_embed[idx][action.parameters.location]
                karel_token_selected_mask[idx][mutation.get_action_name_id(action.parameters.token)] = 1.
                karel_token_mask[idx] = 1.

            if action.id == mutation.WRAP_BLOCK:
                block_type_mask[idx] = 1.
                block_type_selected_mask[idx][mutation.get_block_type_id(action.parameters.block_type)] = 1.

                double_loc_0[idx] = location_embed[idx][action.parameters.start]
                double_loc_1[idx] = location_embed[idx][action.parameters.end]
                double_loc_mask[idx] = 1.

                if action.id < 2:  # `if` | `while`
                    condition_mask[idx] = 1.
                    condition_selected_mask[idx][action.parameters.cond_id] = 1.
                else:  # `repeat`
                    repeat_count_mask[idx] = 1.
                    repeat_count_selected_mask[idx][action.parameters.cond_id] = 1.

            if action.id == mutation.WRAP_IFELSE:
                condition_mask[idx] = 1.
                condition_selected_mask[idx][action.parameters.cond_id] = 1.
                block_type_selected_mask[idx][0] = 1.  # `if`

                triple_loc_0[idx] = location_embed[idx][action.parameters.if_start]
                triple_loc_1[idx] = location_embed[idx][action.parameters.else_start]
                triple_loc_2[idx] = location_embed[idx][action.parameters.end]
                triple_loc_mask[idx] = 1.

        condition_reward = self.model.condition_id(state_enc, block_type_selected_mask)

        batch_location_value = self.model.single_location(location_embed)
        batch_karel_token_value = self.model.karel_token(actions_categorical, state_enc, karel_token_locations)

        double_loc_value = self.model.double_location(double_loc_0, double_loc_1)
        triple_loc_value = self.model.triple_location(triple_loc_0, triple_loc_1, triple_loc_2)

        loss_0 = self.criterion(batch_action_value, targets)

        loss_1 = self.criterion(batch_location_value, targets, batch_location_mask)

        loss_2 = self.criterion((batch_karel_token_value * karel_token_selected_mask).sum(dim=1),
                                targets, karel_token_mask)

        loss_3 = self.criterion((block_type_reward * block_type_selected_mask).sum(dim=1),
                                targets, block_type_mask)

        loss_4 = self.criterion((condition_reward * condition_selected_mask).sum(dim=1),
                                targets, condition_mask)

        loss_5 = self.criterion((repeat_count_reward * repeat_count_selected_mask).sum(dim=1),
                                targets, repeat_count_mask)

        loss_6 = self.criterion(double_loc_value, targets, double_loc_mask)

        loss_7 = self.criterion(triple_loc_value, targets, triple_loc_mask)

        _checkpoint_train = time.clock()

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
        loss.backward()
        self.optimizer.step()

        _end_train = time.clock()

        logger_train.info(f"Train on batch of size: {batch_size}")
        logger_train.info(f"Building batch: {round(_checkpoint_train - _begin_train,3)}s")
        logger_train.info(f"Updating gradients: {round(_end_train - _checkpoint_train,3)}s")
        logger_train.info(f"Total time: {round(_end_train - _begin_train,3)}s")

    def update(self, other):
        self.model.load_state_dict(other.model.state_dict())
