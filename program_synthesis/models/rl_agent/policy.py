import torch
import torch.nn.functional as F
from torch import nn as nn

from program_synthesis.models.modules import karel_common, karel
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, TASK_EMBED_SIZE, LOCATION_EMBED_SIZE, \
    STATE_EMBED_SIZE, KAREL_STATIC_TOKEN, TOKEN_EMBED_SIZE


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

        self.code_token_embed = None
        self.code_enc = None
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
        self.code_token_embed, self.code_enc = self.encode_code(code)
        self._state = torch.cat([task_enc, self.code_enc], dim=1)

        act = self.action_type(self._state)
        return act

    def forward(self):
        raise ValueError("This module can't be evaluated")


class ActionTypeModel(nn.Module):
    """ Compute action value

        Input:
            E:  Embed of the (task, code) [shape: (None, INPUT_EMBED_SIZE)]

        Output:
            A:  Expected reward given an action
    """
    HIDDEN_0 = 32

    def __init__(self):
        super(ActionTypeModel, self).__init__()
        self.dense0 = nn.Linear(STATE_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, TOTAL_MUTATION_ACTIONS)

    def forward(self, embed):
        hidden = F.relu(self.dense0(embed))
        out = F.sigmoid(self.dense1(hidden))
        return out


class LocationParameterModel(nn.Module):
    """ Compute location parameter/value of a given action.
        It is a recurrent nn to support code sequence of different lengths

        Input:
            A:  Action to be done [shape: (BATCH_SIZE, TOTAL_MUTATION_ACTION)]
            E:  Embed of the task [shape: (BATCH_SIZE, TASK_EMBED_SIZE)]
            C:  Code represented as token sequence [shape: (SEQ_LENGTH, BATCH_SIZE, TOKEN_EMBED)]

        Output:
            O:  np.array [shape: (None, SEQ_LENGTH, 1)]
                Action value expected reward
            L:  Embedding for each location [shape: (None, SEQ_LENGTH, LOCATION_EMBED_SIZE)]

        Note:
            For different actions the mask may vary
    """

    def __init__(self):
        super(LocationParameterModel, self).__init__()
        self.lstm = nn.LSTM(TOKEN_EMBED_SIZE, LOCATION_EMBED_SIZE)

        self.dense_cell = nn.Linear(TOTAL_MUTATION_ACTIONS + TASK_EMBED_SIZE, LOCATION_EMBED_SIZE)
        self.dense_reward = nn.Linear(LOCATION_EMBED_SIZE, 1)

    def forward(self, action, tasks_enc, code):
        # Prepare code vector
        # code = code.permute((1, 0, 2))  # Convert to: (Seq length, batch size, vector size)

        seq_length, batch_size, _ = code.shape

        cell = F.relu(self.dense_cell(torch.cat([action, tasks_enc], dim=1)))
        hidden = torch.zeros(*cell.shape)

        comp = (hidden.view(1, *hidden.shape), cell.view(1, *cell.shape))

        loc_embed, _ = self.lstm(code, comp)

        reward = F.sigmoid(self.dense_reward(loc_embed.view(-1, LOCATION_EMBED_SIZE))).view(seq_length, batch_size, -1)

        return reward.permute((1, 0, 2)), loc_embed.permute((1, 0, 2))


class KarelTokenParameterModel(nn.Module):
    """ Compute Karel Token parameter/value of a given action + locations

        Input:
            A: Action (One hot encoding)
            E: Embed of the task
            L: Location Embedding

        Output:
            T: Token one hot encoding
                + move
                + turnLeft
                + turnRight
                + putMarker
                + pickMarker
    """
    HIDDEN_0 = 32
    HIDDEN_1 = 32

    def __init__(self):
        super(KarelTokenParameterModel, self).__init__()
        self.dense0 = nn.Linear(TOTAL_MUTATION_ACTIONS + TASK_EMBED_SIZE + LOCATION_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, self.HIDDEN_1)
        self.dense2 = nn.Linear(self.HIDDEN_1, KAREL_STATIC_TOKEN)

    def forward(self, action, embed, loc_embed):
        repr = torch.cat([action, embed, loc_embed], dim=1)
        hidden = F.relu(self.dense0(repr))
        hidden = F.relu(self.dense1(hidden))
        output = F.softmax(self.dense2(hidden), dim=1)
        return output

