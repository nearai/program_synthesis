"""

Notes:
    Position: 512
    Task: 512
    Code: 512
    State embed size: 1024
        (Concatenation of
            task embed: 512
            code embed: 512)

    Total operations:
        6 (Refer to program_synthesis.karel.dataset.mutation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from program_synthesis.tools import saver
from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.models.modules import karel
from program_synthesis.karel.models.modules import karel_common


class KarelEditPolicy(nn.Module):
    def __init__(self, vocab_size: int, args):
        super(KarelEditPolicy).__init__()

        # Save partial computations of tensors
        self.partial = saver.ArgsDict()

        # Coding input
        self.task_encoder = karel_common.make_task_encoder(args)
        self.code_encoder = karel.CodeEncoderRL(args)
        self.token_embed = nn.Embedding(vocab_size, 256)

        # Getting output
        self.operation_type = OperationTypeModel()

        self.add_action = AddActionModel()
        self.remove_action = RemoveActionModel()
        self.replace_action = AddActionModel()  # Same parameters same model
        self.unwrap_action = RemoveActionModel()  # Same parameters same model
        self.wrap_block = WrapBlockModel()
        self.wrap_ifelse = WrapIfElseModel()

    def encode_task(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        task_encoded = self.task_encoder(input, output)
        self.partial.task_encoded = task_encoded.max(0)[0]
        return self.partial.task_encoded

    def embed_code(self, code: torch.Tensor):
        code = code.type(torch.LongTensor)
        self.partial.code_embed = self.token_embed(code)
        return self.partial.code_embed

    def encode_code(self, code: torch.Tensor):
        """
            `code` expected shape:
                (batch_size x seq_length)
        """
        tokens_embed = self.embed_code(code)

        # `seq` is a semantic embed of each token
        self.partial.position_embed, self.partial.code_embed = self.code_encoder(tokens_embed)
        return self.partial.position_embed, self.partial.code_embed

    def clear(self):
        self.partial.clear()

    def forward(self, *input):
        raise ValueError("This module is not `evaluable`")


class OperationTypeModel(nn.Module):
    """ Compute action value

        Input:
            `code` expected shape:
                (batch_size x state_embed_size)

        Output:
            `out` expected shape:
                (batch_size x total_operations)
    """

    def __init__(self):
        super(OperationTypeModel, self).__init__()
        self.dense0 = nn.Linear(1024, 256)
        self.dense1 = nn.Linear(256, mutation.Operation.total())

    def forward(self, embed):
        hidden = F.relu(self.dense0(embed))
        out = self.dense1(hidden)
        return out


class AddActionModel(nn.Module):
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
        x = torch.cat([position, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x


class RemoveActionModel(nn.Module):
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
        super(RemoveActionModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512, 32)
        self.linear1 = nn.Linear(32, 1)

    def forward(self, position, task):
        x = torch.cat([position, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x


class WrapBlockModel(nn.Module):
    def __init__(self):
        super(WrapBlockModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512 + 512, 256)

        self.head_repeat = nn.Linear(256, len(mutation.REPEAT_COUNTS))
        self.head_if = nn.Linear(256, len(mutation.CONDS))
        self.head_while = nn.Linear(256, len(mutation.CONDS))

    def forward(self, start_pos, end_pos, task):
        x = torch.cat([start_pos, end_pos, task], dim=1)
        x = F.relu(self.linear0(x))

        x_repeat = self.head_repeat(x)
        x_if = self.head_if(x)
        x_while = self.head_while(x)

        return x_repeat, x_if, x_while


class WrapIfElseModel(nn.Module):
    def __init__(self):
        super(WrapIfElseModel, self).__init__()
        self.linear0 = nn.Linear(512 + 512 + 512 + 512, 256)
        self.head = nn.Linear(256, len(mutation.CONDS))

    def forward(self, if_pos, else_pos, end_pos, task):
        x = torch.cat([if_pos, else_pos, end_pos, task], dim=1)
        x = F.relu(self.linear0(x))
        x = self.head(x)
        return x
