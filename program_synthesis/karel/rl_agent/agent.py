""" Karel Agent - Reinforcement Learning

    More information on program_synthesis/models/rl_agent/description.md
"""

import numpy as np
import torch.nn.functional as F
import torch.optim

from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.dataset.refine_env import MutationActionSpace, Action
from program_synthesis.karel.rl_agent import utils
from program_synthesis.karel.rl_agent.policy import KarelEditPolicy


class KarelAgent:
    def __init__(self, vocab, args):
        self.args = args
        self.vocab = vocab

        self.train_mode = False
        self.model = KarelEditPolicy(len(self.vocab), args)

        self._criterion = F.mse_loss
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self._current_task_enc = None

    def set_task(self, task: utils.Task):
        assert self.train_mode is False, "You can't provide task beforehand while training"
        self._current_task_enc = self.model.encode_task(*task)

    def get_task_enc(self, task: utils.Task = None):
        if task is None:
            return self._current_task_enc
        else:
            return self.model.encode_task(*task)

    def action_value(self, codes: torch.LongTensor, tasks: utils.Task = None):
        """ Determine the value of each action (without parameters)

            `codes` expected shape:
                (batch_size x seq_length)
        """
        assert len(codes.size()) == 0, f"`codes` expected 2 dimensions, found {len(codes.size())}"

        if tasks is None:
            assert not self.train_mode
            batch_size = codes.size()[0]
            tasks_enc = self._current_task_enc.repeat(batch_size, 1)
        else:
            tasks_enc = self.model.encode_task(*tasks)

        _, codes_enc = self.model.encode_code(codes)

        assert tasks.size()[0] == codes_enc.size()[0], "Batch size dimension must coincide"
        return self.model.operation_type(tasks_enc, codes_enc)

    def best_action_value(self, codes: torch.LongTensor, tasks: utils.Task = None) -> (torch.Tensor, torch.Tensor):
        action_values = self.action_value(codes, tasks)
        return action_values.max(dim=1)  # (max, argmax)

    def select_action(self, code):
        assert self._current_task_enc is not None

        action_space = MutationActionSpace(code=code)

        code = utils.prepare_code(action_space.atree.code, self.vocab, tensor=True)
        position_enc, code_enc = self.model.encode_code(code)
        task_enc = self._current_task_enc

        with torch.no_grad():
            while True:
                if np.random.random() < self.args.rl_eps_action:
                    action = action_space.sample()
                    break
                else:
                    action_type = int(self.model.operation_type(task_enc, code_enc).argmax())

                    if np.random.random() < self.args.rl_eps_parameter:
                        action_params = action_space.sample_parameters(action_type)
                        if action_params is None:
                            continue
                    else:
                        parameters = list(action_space.valid_parameters_locations(action_type))

                        if len(parameters) == 0:
                            action_params = None
                        else:
                            valid_params, value = ParameterSelector.get_parameters_value(action_type, self.model,
                                                                                         parameters, position_enc,
                                                                                         task_enc)
                            ix = value.argmax()
                            action_params = valid_params[ix]

                    if action_params is None:
                        continue

                    action = Action(action_type, action_params)
                    break

        return action

    def update(self, other):
        self.model.load_state_dict(other.model.state_dict())

    def set_train(self, mode):
        self.train_mode = mode
        self.model.train(mode)

        if mode:  # Training mode
            self._current_task_enc = None
        else:  # Testing mode
            pass


class ParameterSelector:
    @staticmethod
    def get_parameters_value(action_type, model, parameters, position_enc, task_enc):
        action_type_func = {
            mutation.ADD_ACTION: ParameterSelector.add_action,
            mutation.REMOVE_ACTION: ParameterSelector.remove_action,
            mutation.REPLACE_ACTION: ParameterSelector.replace_action,
            mutation.UNWRAP_BLOCK: ParameterSelector.unwrap_block,
            mutation.WRAP_BLOCK: ParameterSelector.wrap_block,
            mutation.WRAP_IFELSE: ParameterSelector.wrap_ifelse,
        }

        assert action_type in action_type_func, \
            f"Expected to be in the range [0, 6) found {action_type} ({type(action_type)})"

        return action_type_func.get(action_type)(model, parameters, position_enc, task_enc)

    @staticmethod
    def add_action(model: KarelEditPolicy, parameters, position_enc, task_enc):
        pos = position_enc[:, parameters].squeeze(0)
        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value = model.add_action(pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, loc in enumerate(parameters):
            for t_ix, tok in enumerate(mutation.ACTION_NAMES):
                params_l.append(mutation.ActionAddParameters(loc, tok))
                value_l.append(value[l_ix][t_ix])

        return params_l, torch.Tensor(value_l)

    @staticmethod
    def remove_action(model: KarelEditPolicy, parameters, position_enc, task_enc):
        pos = position_enc[:, parameters].squeeze(0)
        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value = model.remove_action(pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, loc in enumerate(parameters):
            params_l.append(mutation.ActionRemoveParameters(loc))
            value_l.append(value[l_ix][0])

        return params_l, torch.Tensor(value_l)

    @staticmethod
    def replace_action(model: KarelEditPolicy, parameters, position_enc, task_enc):
        pos = position_enc[:, parameters].squeeze(0)
        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value = model.replace_action(pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, loc in enumerate(parameters):
            for t_ix, tok in enumerate(mutation.ACTION_NAMES):
                params_l.append(mutation.ActionReplaceParameters(loc, tok))
                value_l.append(value[l_ix][t_ix])

        return params_l, torch.Tensor(value_l)

    @staticmethod
    def unwrap_block(model: KarelEditPolicy, parameters, position_enc, task_enc):
        pos = position_enc[:, parameters].squeeze(0)
        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value = model.unwrap_block(pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, loc in enumerate(parameters):
            params_l.append(mutation.ActionUnwrapBlockParameters(loc))
            value_l.append(value[l_ix][0])

        return params_l, torch.Tensor(value_l)

    @staticmethod
    def wrap_block(model: KarelEditPolicy, parameters, position_enc, task_enc):
        start_pos = position_enc[:, [loc[0] for loc in parameters]].squeeze(0)
        end_pos = position_enc[:, [loc[1] for loc in parameters]].squeeze(0)

        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value_repeat, value_if, value_while = model.wrap_block(start_pos, end_pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, (s_loc, e_loc) in enumerate(parameters):
            for c_ix, _ in enumerate(mutation.REPEAT_COUNTS):  # Repeat
                params_l.append(
                    mutation.ActionWrapBlockParameters(mutation.BLOCK_TYPE[-1], c_ix, s_loc, e_loc))
                value_l.append(value_repeat[l_ix][c_ix])

            for c_ix, _ in enumerate(mutation.CONDS):  # If
                params_l.append(
                    mutation.ActionWrapBlockParameters(mutation.BLOCK_TYPE[0], c_ix, s_loc, e_loc))
                value_l.append(value_if[l_ix][c_ix])

            for c_ix, _ in enumerate(mutation.CONDS):  # While
                params_l.append(
                    mutation.ActionWrapBlockParameters(mutation.BLOCK_TYPE[1], c_ix, s_loc, e_loc))
                value_l.append(value_while[l_ix][c_ix])

        return params_l, torch.Tensor(value_l)

    @staticmethod
    def wrap_ifelse(model: KarelEditPolicy, parameters, position_enc, task_enc):
        if_pos = position_enc[:, [loc[0] for loc in parameters]].squeeze(0)
        else_pos = position_enc[:, [loc[1] for loc in parameters]].squeeze(0)
        end_pos = position_enc[:, [loc[2] for loc in parameters]].squeeze(0)

        batch_size = len(parameters)
        task_enc = task_enc.repeat(batch_size, 1)

        value = model.wrap_ifelse(if_pos, else_pos, end_pos, task_enc)

        params_l = []
        value_l = []

        for l_ix, locs in enumerate(parameters):
            for c_ix, _ in enumerate(mutation.CONDS):  # If
                params_l.append(
                    mutation.ActionWrapIfElseParameters(c_ix, locs[0], locs[1], locs[2]))
                value_l.append(value[l_ix][c_ix])

        return params_l, torch.Tensor(value_l)
