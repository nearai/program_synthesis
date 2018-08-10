import torch
import torch.nn.functional as F
from torch import nn as nn

from program_synthesis.karel.models.modules import karel_common, karel
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, LOCATION_EMBED_SIZE, \
    STATE_EMBED_SIZE, KAREL_STATIC_TOKEN, TOKEN_EMBED_SIZE, BLOCK_TYPE_SIZE, CONDITION_SIZE, REPEAT_COUNT_SIZE


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
        self.single_location = SingleLocationModel()
        self.double_location = DoubleLocationModel()
        self.triple_location = TripleLocationModel()
        self.block_type = BlockTypeModel()
        self.condition_id = ConditionIdModel()
        self.repeat_count = RepeatCountModel()

        self.code_token_embed = None
        self.code_enc = None
        self.state_enc = None

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
        self.state_enc = torch.cat([task_enc, self.code_enc], dim=1)

        act = self.action_type(self.state_enc)
        return act

    def forward(self):
        raise ValueError("This module can't be evaluated")


class ActionTypeModel(nn.Module):
    """ Compute action value

        Input:
            E:  Embed of the (task, code) [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]

        Output:
            A:  Expected reward given an action [shape: (BATCH_SIZE, TOTAL_MUTATION_ACTIONS)]
    """
    HIDDEN_0 = 32

    def __init__(self):
        super(ActionTypeModel, self).__init__()
        self.dense0 = nn.Linear(STATE_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, TOTAL_MUTATION_ACTIONS)

    def forward(self, embed):
        hidden = F.relu(self.dense0(embed))
        out = self.dense1(hidden)
        return out


class LocationParameterModel(nn.Module):
    """ Compute location parameter/value of a given action.
        It is a recurrent nn to support code sequence of different lengths

        Input:
            A:  Action to be done [shape: (BATCH_SIZE, TOTAL_MUTATION_ACTION)]
            E:  Embed of the state [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]
            C:  Code represented as token sequence [shape: (SEQ_LENGTH, BATCH_SIZE, TOKEN_EMBED)]

        Output:
            L:  Embedding for each location [shape: (BATCH_SIZE, SEQ_LENGTH, LOCATION_EMBED_SIZE)]
    """

    def __init__(self):
        super(LocationParameterModel, self).__init__()
        self.lstm = nn.LSTM(TOKEN_EMBED_SIZE, LOCATION_EMBED_SIZE)
        self.dense_cell = nn.Linear(TOTAL_MUTATION_ACTIONS + STATE_EMBED_SIZE, LOCATION_EMBED_SIZE)

    def forward(self, action, state_enc, code):
        # Prepare code vector
        # code = code.permute((1, 0, 2))  # Convert to: (Seq length, batch size, vector size)

        seq_length, batch_size, _ = code.shape

        cell = F.relu(self.dense_cell(torch.cat([action, state_enc], dim=1)))
        hidden = torch.zeros(*cell.shape)

        comp = (hidden.view(1, *hidden.shape), cell.view(1, *cell.shape))

        loc_embed, _ = self.lstm(code, comp)

        return loc_embed.permute((1, 0, 2))


class SingleLocationModel(nn.Module):
    """ Predict parameter location of actions that require single location
        such as `ADD`, `REMOVE`, `REPLACE`, `UNWRAP`

        Input:
            L:  Embedding for each location [shape: (*DIMS, LOCATION_EMBED_SIZE)]

        Output:
            O:  Action value expected reward [shape: (*DIMS, 1)]
    """

    def __init__(self):
        super(SingleLocationModel, self).__init__()
        self.dense_reward = nn.Linear(LOCATION_EMBED_SIZE, 1)

    def forward(self, loc_embed):
        shape = loc_embed.shape
        loc_embed = loc_embed.contiguous().view(-1, LOCATION_EMBED_SIZE)
        reward = self.dense_reward(loc_embed).view(*shape[:-1], -1)

        return reward


class DoubleLocationModel(nn.Module):
    """ Predict parameter location of actions that require two location
        such as `WRAP`

        Input:
            L0:  Embedding for each first location [shape: (*DIM, LOCATION_EMBED_SIZE)]
            L1:  Embedding for each first location [shape: (*DIM, LOCATION_EMBED_SIZE)]

        Output:
            O:  Action value expected reward [shape: (*DIM, 1)]
    """

    def __init__(self):
        super(DoubleLocationModel, self).__init__()
        self.dense_reward = nn.Linear(2 * LOCATION_EMBED_SIZE, 1)

    def forward(self, loc0_embed, loc1_embed):
        shape = loc0_embed.shape

        loc0_embed = loc0_embed.contiguous().view(-1, LOCATION_EMBED_SIZE)
        loc1_embed = loc1_embed.contiguous().view(-1, LOCATION_EMBED_SIZE)

        L = torch.cat([loc0_embed, loc1_embed], dim=1)

        reward = self.dense_reward(L).view(*shape[:-1], -1)

        return reward


class TripleLocationModel(nn.Module):
    """ Predict parameter location of actions that require three location
        such as `ADD_IF_ELSE`

        Input:
            L0:  Embedding for each first location [shape: (*DIM, LOCATION_EMBED_SIZE)]
            L1:  Embedding for each first location [shape: (*DIM, LOCATION_EMBED_SIZE)]
            L2:  Embedding for each first location [shape: (*DIM, LOCATION_EMBED_SIZE)]

        Output:
            O:  Action value expected reward [shape: (*DIM, 1)]
    """

    def __init__(self):
        super(TripleLocationModel, self).__init__()
        self.dense_reward = nn.Linear(3 * LOCATION_EMBED_SIZE, 1)

    def forward(self, loc_embed0, loc_embed1, loc_embed2):
        shape = loc_embed0.shape

        loc_embed0 = loc_embed0.contiguous().view(-1, LOCATION_EMBED_SIZE)
        loc_embed1 = loc_embed1.contiguous().view(-1, LOCATION_EMBED_SIZE)
        loc_embed2 = loc_embed2.contiguous().view(-1, LOCATION_EMBED_SIZE)

        L = torch.cat([loc_embed0, loc_embed1, loc_embed2], dim=1)

        reward = self.dense_reward(L).view(*shape[:-1], -1)

        return reward


class KarelTokenParameterModel(nn.Module):
    """ Compute Karel Token parameter/value of a given action + locations

        Input:
            A: Action (One hot encoding) [shape: (BATCH_SIZE, TOTAL_MUTATION_ACTIONS)]
            E: Embed of the state [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]
            L: Location Embedding [shape: (BATCH_SIZE, LOCATION_EMBED_SIZE)]

        Output:
            T: Token one hot encoding [shape: (BATCH_SIZE, KAREL_STATIC_TOKEN)]
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
        self.dense0 = nn.Linear(TOTAL_MUTATION_ACTIONS + STATE_EMBED_SIZE + LOCATION_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, self.HIDDEN_1)
        self.dense2 = nn.Linear(self.HIDDEN_1, KAREL_STATIC_TOKEN)

    def forward(self, action, embed, loc_embed):
        repr = torch.cat([action, embed, loc_embed], dim=1)
        hidden = F.relu(self.dense0(repr))
        hidden = F.relu(self.dense1(hidden))
        output = self.dense2(hidden)
        return output


class BlockTypeModel(nn.Module):
    """ Compute block type parameter/value

        Input:
            E: Embed of the state [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]

        Output:
            T: Token one hot encoding [shape: (BATCH_SIZE, BLOCK_TYPE_SIZE)]
                + if
                + while
                + repeat
    """
    HIDDEN_0 = 32
    HIDDEN_1 = 32

    def __init__(self):
        super(BlockTypeModel, self).__init__()
        self.dense0 = nn.Linear(STATE_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, self.HIDDEN_1)
        self.dense2 = nn.Linear(self.HIDDEN_1, BLOCK_TYPE_SIZE)

    def forward(self, state_embed):
        hidden = F.relu(self.dense0(state_embed))
        hidden = F.relu(self.dense1(hidden))
        out = self.dense2(hidden)
        return out


class ConditionIdModel(nn.Module):
    """ Compute block type parameter/value

        Input:
            E: Embed of the state [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]
            B: Block type [shape: (BATCH_SIZE, BLOCK_TYPE_SIZE)]
                [*] if
                [*] while
                [ ] repeat

        Output:
            T: Token one hot encoding [shape: (BATCH_SIZE, CONDITION_SIZE)]
                + frontIsClear
                + leftIsClear
                + ...
    """
    HIDDEN_0 = 32
    HIDDEN_1 = 32

    def __init__(self):
        super(ConditionIdModel, self).__init__()
        self.dense0 = nn.Linear(STATE_EMBED_SIZE + BLOCK_TYPE_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, self.HIDDEN_1)
        self.dense2 = nn.Linear(self.HIDDEN_1, CONDITION_SIZE)

    def forward(self, state_embed, block_type):
        inp = torch.cat([state_embed, block_type], dim=1)
        hidden = F.relu(self.dense0(inp))
        hidden = F.relu(self.dense1(hidden))
        out = self.dense2(hidden)
        return out


class RepeatCountModel(nn.Module):
    """ Compute block type parameter/value

        Input:
            E: Embed of the state [shape: (BATCH_SIZE, STATE_EMBED_SIZE)]

        Output:
            T: Token one hot encoding [shape: (BATCH_SIZE, REPEAT_COUNT_SIZE)]
                + 2
                + 3
                + ...
                + 10
    """
    HIDDEN_0 = 32
    HIDDEN_1 = 32

    def __init__(self):
        super(RepeatCountModel, self).__init__()
        self.dense0 = nn.Linear(STATE_EMBED_SIZE, self.HIDDEN_0)
        self.dense1 = nn.Linear(self.HIDDEN_0, self.HIDDEN_1)
        self.dense2 = nn.Linear(self.HIDDEN_1, REPEAT_COUNT_SIZE)

    def forward(self, embed):
        hidden = F.relu(self.dense0(embed))
        hidden = F.relu(self.dense1(hidden))
        out = self.dense2(hidden)
        return out
