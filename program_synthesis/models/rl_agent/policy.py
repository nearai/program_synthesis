import torch
import torch.nn.functional as F
from torch import nn as nn

from program_synthesis.datasets.karel import mutation
from program_synthesis.models.modules import karel_common, karel
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, TASK_EMBED_SIZE, LOCATION_EMBED_SIZE, \
    TOKEN_EMBED_SIZE


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


class ActionTypeModel(nn.Module):
    """ Compute action value

        Input:
            E:  Embed of the (task, code) [shape: (None, INPUT_EMBED_SIZE)]

        Output:
            A:  Expected reward given an action
    """

    def __init__(self):
        super(ActionTypeModel, self).__init__()
        self.dense0 = nn.Linear(TASK_EMBED_SIZE, 20)
        self.dense1 = nn.Linear(20, TOTAL_MUTATION_ACTIONS)

    def forward(self, embed):
        hidden = F.relu(self.dense0(embed))
        out = F.sigmoid(self.dense1(hidden))
        return out


class LocationParameterModel(nn.Module):
    """ Compute location parameter/value of a given action.
        It is a recurrent nn to support code sequence of different lengths

        Input:
            A:  Action to be done [shape: (None, TOTAL_MUTATION_ACTION)]
            E:  Embed of the (task, code) [shape: (None, INPUT_EMBED_SIZE)]
            C:  np.array Code represented as token sequence [shape: (None, SEQ_LENGTH, TOKEN_EMBED)]
            M:  np.array
                M.shape == C.shape
                Mask array denoting that an action in the given location is valid.

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

    def forward(self, action, embed, code, mask):
        # Prepare code vector
        code = code.permute((1, 0, 2))  # Seq length, batch size, vector size

        seq_length, batch_size, _ = code.shape

        cell = F.relu(self.dense_cell(torch.cat([action, embed], dim=1)))
        hidden = torch.zeros(cell.shape)

        comp = (hidden.view(1, *hidden.shape), cell.view(1, *cell.shape))

        loc_embed, _ = self.lstm(code, comp)

        reward = F.sigmoid(self.dense_reward(loc_embed.view(-1, LOCATION_EMBED_SIZE))).view(seq_length, batch_size, -1)
        reward = reward * mask.permute((1, 0)).view(seq_length, batch_size, 1)

        return reward.permute((1, 0, 2)), loc_embed.permute((1, 0, 2))


class KarelTokenParameterModel(nn.Module):
    """ Compute Karel Token parameter/value of a given action + locations

        Input:
            A: Action (One hot encoding)
            E: Embed of the (task, code)
            L: Location Embedding

        Output:
            T: Token one hot encoding
                + move
                + turnLeft
                + turnRight
                + putMarker
                + pickMarker
    """

    def __init__(self):
        super(KarelTokenParameterModel, self).__init__()
        self.dense0 = nn.Linear(TOTAL_MUTATION_ACTIONS + TASK_EMBED_SIZE + LOCATION_EMBED_SIZE, 50)
        self.dense1 = nn.Linear(50, 50)
        self.dense2 = nn.Linear(50, 5)

    def forward(self, action, embed, loc_embed):
        repr = torch.cat([action, embed, loc_embed], dim=1)
        hidden = F.relu(self.dense0(repr))
        hidden = F.relu(self.dense1(hidden))
        output = F.softmax(self.dense2(hidden))
        return output
