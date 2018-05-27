import torch

from program_synthesis.models.rl_agent.policy import LocationParameterModel, KarelTokenParameterModel, \
    LOCATION_EMBED_SIZE
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, TASK_EMBED_SIZE, VOCAB_SIZE, \
    MAX_TOKEN_PER_CODE, TOKEN_EMBED_SIZE


def test_location_parameter():
    batch_size = 3
    code_length = MAX_TOKEN_PER_CODE

    action = torch.randn(batch_size, TOTAL_MUTATION_ACTIONS)
    embed = torch.randn(batch_size, TASK_EMBED_SIZE)

    code = torch.randn(code_length, batch_size, TOKEN_EMBED_SIZE)

    model = LocationParameterModel()

    # print(action.shape)
    # print(embed.shape)
    # print(code.shape)
    # print(mask.shape)
    reward, loc_embed = model(action, embed, code)

    print("REWARD:", reward.shape)
    print("LOCATION_EMBEDDING:", loc_embed.shape)


def test_karel_token():
    size = 3

    model = KarelTokenParameterModel()

    x = model(
        torch.randn(size, TOTAL_MUTATION_ACTIONS),
        torch.randn(size, TASK_EMBED_SIZE),
        torch.randn(size, LOCATION_EMBED_SIZE)
    )

    print(x)
    print(x.sum(dim=1))


if __name__ == '__main__':
    test_location_parameter()
