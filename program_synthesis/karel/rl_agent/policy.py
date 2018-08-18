"""

Notes:
    Position: 512
    Task: 512
    Code: 256
    State embed size: 768
        (Concatenation of
            task embed: 512
            code embed: 256)

    Total operations:
        6 (Refer to program_synthesis.karel.dataset.mutation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.models.modules import karel
from program_synthesis.karel.models.modules import karel_common


class KarelEditPolicy(nn.Module):

    def __init__(self, vocab_size: int, args):
        super(KarelEditPolicy, self).__init__()

        # Coding input
        self.task_encoder = karel_common.make_task_encoder(args)
        self.code_encoder = karel.CodeEncoderRL(args)
        self.token_embed = nn.Embedding(vocab_size, 256)

        # Getting output
        self.operation_type = OperationTypeModel()

        self.add_action = AddActionModel()
        self.remove_action = RemoveActionModel()
        self.replace_action = AddActionModel()  # Same parameters same model
        self.unwrap_block = RemoveActionModel()  # Same parameters same model
        self.wrap_block = WrapBlockModel()
        self.wrap_ifelse = WrapIfElseModel()

        self._action2model = {
            mutation.ADD_ACTION: self.add_action,
            mutation.REMOVE_ACTION: self.remove_action,
            mutation.REPLACE_ACTION: self.replace_action,
            mutation.UNWRAP_BLOCK: self.unwrap_block,
            mutation.WRAP_BLOCK: self.wrap_block,
            mutation.WRAP_IFELSE: self.wrap_ifelse
        }

        if not args.train_task_encoder:
            for params in self.task_encoder.parameters():
                params.requires_grad_(False)

    def grad_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def get_parameters_tensors(self, codes_enc, tasks_enc, actions):
        param_value = []

        for idx, action in enumerate(actions):
            model = self.get_model_from_action(action.id)

            if action.id == mutation.ADD_ACTION:
                values = model(codes_enc[None, idx, action.parameters.location], tasks_enc[None, idx])
                value = values[:, mutation.get_action_name_id(action.parameters.token)]

            elif action.id == mutation.REMOVE_ACTION:
                values = model(codes_enc[None, idx, action.parameters.location], tasks_enc[None, idx])
                value = values[:, 0]

            elif action.id == mutation.REPLACE_ACTION:
                values = model(codes_enc[None, idx, action.parameters.location], tasks_enc[None, idx])
                value = values[:, mutation.get_action_name_id(action.parameters.token)]

            elif action.id == mutation.UNWRAP_BLOCK:
                values = model(codes_enc[None, idx, action.parameters.location], tasks_enc[None, idx])
                value = values[:, 0]

            elif action.id == mutation.WRAP_BLOCK:
                values = model(codes_enc[None, idx, action.parameters.start],
                               codes_enc[None, idx, action.parameters.end],
                               tasks_enc[None, idx])
                value = values[mutation.get_block_type_id(action.parameters.block_type)][:, action.parameters.cond_id]

            elif action.id == mutation.WRAP_IFELSE:
                values = model(codes_enc[None, idx, action.parameters.if_start],
                               codes_enc[None, idx, action.parameters.else_start],
                               codes_enc[None, idx, action.parameters.end],
                               tasks_enc[None, idx])

                value = values[:, action.parameters.cond_id]
            else:
                raise ValueError(f"action.id must lie in the range [0, 8) found {action.id}")

            param_value.append(value)

        assert len(param_value) == len(actions)

        return torch.cat(param_value)

    def get_model_from_action(self, action_type):
        return self._action2model.get(action_type)

    def encode_task(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
            `input` & `output` expected shape:
                (batch_size x num_examples x 15 x 18 x 18)
        """
        batch_size, num_examples, _, _, _ = input.size()

        input = input.view(-1, 15, 18, 18)
        output = output.view(-1, 15, 18, 18)

        task_encoded = self.task_encoder(input, output)
        task_encoded = task_encoded.view(batch_size, num_examples, -1)
        task_encoded = task_encoded.max(1)[0]

        return task_encoded

    def embed_code(self, code: torch.Tensor):
        code = code.type(torch.LongTensor)
        code_embed = self.token_embed(code)
        return code_embed

    def encode_code(self, code: torch.Tensor):
        """
            `code` expected shape:
                (batch_size x seq_length)
        """
        tokens_embed = self.embed_code(code)

        # `position_embed` is a "logic" embed of each token
        position_embed, code_embed = self.code_encoder(tokens_embed)
        return position_embed, code_embed

    def forward(self, *input):
        raise ValueError("This module is not `evaluable`")


class BaseModel(nn.Module):
    @staticmethod
    def _assert_size(param, size):
        if isinstance(size, tuple):
            assert len(param.size()) == len(size), \
                f"Expected tensor of dimension {size} found {param.size()}"

            for ps, rs in zip(param.size(), size):
                assert rs is None or ps == rs, \
                    f"Expected tensor of dimension {size} found {param.size()}"

        else:
            assert len(param.size()) == size, \
                f"Expected tensor of dimension {size} found {len(param.size())} ({param.size()})"

    @staticmethod
    def _assert_batch_size(*params):
        batch_size = params[0].size()[0]
        ok = True

        for p in params:
            if p.size()[0] != batch_size:
                ok = False
                break

        assert ok, f"Expected first dimension to coincide but found {[p.size() for p in params]}"


class OperationTypeModel(BaseModel):
    """ Compute action value

        Input:
            `task` expected shape:
                (batch_size x task_embed_size)

            `code` expected shape:
                (batch_size x code_embed_size)

        Output:
            `out` expected shape:
                (batch_size x total_operations)
    """

    def __init__(self):
        super(OperationTypeModel, self).__init__()
        self.dense0 = nn.Linear(512 + 256, 256)
        self.dense1 = nn.Linear(256, 6)

    def forward(self, task, code):
        self.check_input(task, code)

        state = torch.cat([task, code], dim=1)
        hidden = F.relu(self.dense0(state))
        out = self.dense1(hidden)
        return out

    def check_input(self, task, code):
        self._assert_size(task, (None, 512))
        self._assert_size(code, (None, 256))
        self._assert_batch_size(task, code)


class AddActionModel(BaseModel):
    """
        Input:
            `position` expected shape:
                (batch_size x position_embed)

            `task` expected shape:
                (batch_size x task_embed[512])


        Output:
            `x` expected shape:
                (batch_size x possible_actions)
    """

    def __init__(self):
        super(AddActionModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512, 32)
        self.linear1 = nn.Linear(32, len(mutation.ACTION_NAMES))

    def forward(self, position, task):
        self.check_input(position, task)

        x = torch.cat([position, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x

    def check_input(self, position, task):
        self._assert_size(position, (None, 512))
        self._assert_size(task, (None, 512))
        self._assert_batch_size(position, task)


class RemoveActionModel(BaseModel):
    """
        Input:
            `position` expected shape:
                (batch_size x position_embed)

            `task` expected shape:
                (batch_size x task_embed[512])

        Output:
            `x` expected shape:
                (batch_size,)
    """

    def __init__(self):
        super(RemoveActionModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512, 32)
        self.linear1 = nn.Linear(32, 1)

    def forward(self, position, task):
        self.check_input(position, task)

        x = torch.cat([position, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x

    def check_input(self, position, task):
        self._assert_size(position, (None, 512))
        self._assert_size(task, (None, 512))
        self._assert_batch_size(position, task)


class WrapBlockModel(BaseModel):
    """
        Input:
            `start_pos` expected shape:
                (batch_size x position_embed)

            `end_pos` expected shape:
                (batch_size x position_embed)

            `task` expected shape:
                (batch_size x task_embed[512])

        Output:
            `x_if` expected shape:
                (batch_size x |mutation.CONDS|)

            `x_while` expected shape:
                (batch_size x |mutation.CONDS|)

            `x_repeat` expected shape:
                (batch_size x |mutation.REPEAT_COUNTS|)
    """

    def __init__(self):
        super(WrapBlockModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512 + 512, 256)

        self.head_if = nn.Linear(256, len(mutation.CONDS))
        self.head_while = nn.Linear(256, len(mutation.CONDS))
        self.head_repeat = nn.Linear(256, len(mutation.REPEAT_COUNTS))

    def forward(self, start_pos, end_pos, task):
        self.check_input(start_pos, end_pos, task)

        x = torch.cat([start_pos, end_pos, task], dim=1)
        x = F.relu(self.linear0(x))

        x_if = self.head_if(x)
        x_while = self.head_while(x)
        x_repeat = self.head_repeat(x)

        return x_if, x_while, x_repeat

    def check_input(self, start_pos, end_pos, task):
        self._assert_size(start_pos, (None, 512))
        self._assert_size(end_pos, (None, 512))
        self._assert_batch_size(start_pos, end_pos, task)


class WrapIfElseModel(BaseModel):
    """
        Input:
            `if_pos` expected shape:
                (batch_size x position_embed)

            `else_pos` expected shape:
                (batch_size x position_embed)

            `end_pos` expected shape:
                (batch_size x position_embed)

            `task` expected shape:
                (batch_size x task_embed[512])

        Output:
            `x` expected shape:
                (batch_size x |mutation.CONDS|)
    """

    def __init__(self):
        super(WrapIfElseModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512 + 512 + 512, 256)
        self.head = nn.Linear(256, len(mutation.CONDS))

    def forward(self, if_pos, else_pos, end_pos, task):
        self.check_input(if_pos, else_pos, end_pos, task)
        x = torch.cat([if_pos, else_pos, end_pos, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.head(x)
        return x

    def check_input(self, if_pos, else_pos, end_pos, task):
        self._assert_size(if_pos, (None, 512))
        self._assert_size(else_pos, (None, 512))
        self._assert_size(end_pos, (None, 512))
        self._assert_batch_size(if_pos, else_pos, end_pos, task)
