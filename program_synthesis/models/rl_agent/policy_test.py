import torch

from program_synthesis.models.rl_agent.policy import LocationParameterModel, KarelTokenParameterModel, \
    SingleLocationModel, DoubleLocationModel, TripleLocationModel
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, TASK_EMBED_SIZE, MAX_TOKEN_PER_CODE, \
    TOKEN_EMBED_SIZE, LOCATION_EMBED_SIZE, STATE_EMBED_SIZE


def test_location_parameter():
    batch_size = 3
    code_length = MAX_TOKEN_PER_CODE

    action = torch.randn(batch_size, TOTAL_MUTATION_ACTIONS)
    embed = torch.randn(batch_size, TASK_EMBED_SIZE)

    code = torch.randn(code_length, batch_size, TOKEN_EMBED_SIZE)

    model = LocationParameterModel()

    loc_embed = model(action, embed, code)

    print("Location embedding:", loc_embed.shape)

    model_single = SingleLocationModel()
    rewards = model_single(loc_embed)

    print("Reward:", rewards.shape)


def test_multiple_location_parameter():
    batch_size = 4
    code_length = 17

    action = torch.randn(batch_size, TOTAL_MUTATION_ACTIONS)
    embed = torch.randn(batch_size, TASK_EMBED_SIZE)

    code = torch.randn(code_length, batch_size, TOKEN_EMBED_SIZE)

    model = LocationParameterModel()
    loc_embed = model(action, embed, code)

    double = DoubleLocationModel()
    triple = TripleLocationModel()

    r2 = double(loc_embed, loc_embed)
    r3 = triple(loc_embed, loc_embed, loc_embed)

    print(r2.shape)
    print(r3.shape)


def test_karel_token():
    size = 3

    model = KarelTokenParameterModel()

    x = model(
        torch.randn(size, TOTAL_MUTATION_ACTIONS),
        torch.randn(size, STATE_EMBED_SIZE),
        torch.randn(size, LOCATION_EMBED_SIZE)
    )

    print(x)
    print(x.sum(dim=1))


if __name__ == '__main__':
    test_multiple_location_parameter()
