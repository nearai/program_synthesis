import torch
import torch.nn as nn
import torch.nn.functional as F

TOTAL_MUTATION_ACTIONS = 8
INPUT_EMBED_SIZE = 1024
LOCATION_EMBED_SIZE = 20

# TODO: Set `VOCAB_SIZE` variable in a configuration file
VOCAB_SIZE = 30
TOKEN_EMBED_SIZE = 256


class ActionTypeModel(nn.Module):
    """ Compute action value

        Input:
            E:  Embed of the (task, code) [shape: (None, INPUT_EMBED_SIZE)]

        Output:
            A:  Expected reward given an action
    """

    def __init__(self):
        super(ActionTypeModel, self).__init__()
        self.dense0 = nn.Linear(INPUT_EMBED_SIZE, 20)
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

        self.dense_cell = nn.Linear(TOTAL_MUTATION_ACTIONS + INPUT_EMBED_SIZE, LOCATION_EMBED_SIZE)
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
        self.dense0 = nn.Linear(TOTAL_MUTATION_ACTIONS + INPUT_EMBED_SIZE + LOCATION_EMBED_SIZE, 50)
        self.dense1 = nn.Linear(50, 50)
        self.dense2 = nn.Linear(50, 5)

    def forward(self, action, embed, loc_embed):
        repr = torch.cat([action, embed, loc_embed], dim=1)
        hidden = F.relu(self.dense0(repr))
        hidden = F.relu(self.dense1(hidden))
        output = F.softmax(self.dense2(hidden))
        return output


def test_karel_token():
    size = 3

    model = KarelTokenParameterModel()

    x = model(
        torch.randn(size, TOTAL_MUTATION_ACTIONS),
        torch.randn(size, INPUT_EMBED_SIZE),
        torch.randn(size, LOCATION_EMBED_SIZE)
    )

    print(x.sum(1))


def test_location_parameter():
    import random

    batch_size = 3
    code_length = random.randint(10, 20)

    action = torch.randn(batch_size, TOTAL_MUTATION_ACTIONS)
    embed = torch.randn(batch_size, INPUT_EMBED_SIZE)

    code = torch.randn(batch_size, code_length, VOCAB_SIZE)
    mask = (torch.randn(batch_size, code_length) > .5).type(torch.FloatTensor)

    model = LocationParameterModel()

    reward, loc_embed = model(action, embed, code, mask)

    print(reward)
    print(loc_embed)


if __name__ == '__main__':
    test_location_parameter()
