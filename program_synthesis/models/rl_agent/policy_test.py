import torch

from program_synthesis.models.rl_agent.policy import LocationParameterModel, KarelTokenParameterModel, \
    LOCATION_EMBED_SIZE
from program_synthesis.models.rl_agent.config import TOTAL_MUTATION_ACTIONS, TASK_EMBED_SIZE, VOCAB_SIZE, \
    MAX_TOKEN_PER_CODE


def test_location_parameter():
    batch_size = 3
    code_length = MAX_TOKEN_PER_CODE

    action = torch.randn(batch_size, TOTAL_MUTATION_ACTIONS)
    embed = torch.randn(batch_size, TASK_EMBED_SIZE)

    code = torch.randn(batch_size, code_length, VOCAB_SIZE)
    mask = (torch.randn(batch_size, code_length) > .5).type(torch.FloatTensor)

    model = LocationParameterModel()

    reward, loc_embed = model(action, embed, code, mask)

    print("REWARD:\n", reward.shape)
    print("LOCATION_EMBEDDING:\n", loc_embed.shape)


def test_karel_token():
    size = 3

    model = KarelTokenParameterModel()

    x = model(
        torch.randn(size, TOTAL_MUTATION_ACTIONS),
        torch.randn(size, TASK_EMBED_SIZE),
        torch.randn(size, LOCATION_EMBED_SIZE)
    )

    print(x.sum(1))


if __name__ == '__main__':
    test_location_parameter()
