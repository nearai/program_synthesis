import torch
from program_synthesis.datasets.karel import mutation

import program_synthesis.datasets.karel.utils as utils
import program_synthesis.models.rl_agent.agent as agent
import program_synthesis.models.rl_agent.environment as karel_env
import program_synthesis.tools.saver as saver
from program_synthesis.datasets.karel.mutation import Action, ActionAddParameters

args = saver.ArgsDict(
    num_epochs=100, max_rollout_length=10, replay_buffer_size=16384, erase_factor=0.01,
    num_episodes=10, num_training_steps=10, batch_size=32, update_actor_epoch=10,
    karel_io_enc='lgrl', lr=0.001, cuda=False)


def show_code(code_tensor, env):
    code_tensor = code_tensor.reshape(-1).type(torch.LongTensor)
    return utils.beautify(env.recover_code(code_tensor))


def test_add_action():
    env = karel_env.KarelEditEnv()
    task, code = env.reset()
    kagent = agent.KarelAgent(env, args)
    kagent.set_task(task)

    EPSILON_GREEDY = 0.

    for i in range(20):
        action = kagent.select_action(code, EPSILON_GREEDY)

        code, reward, done, info = env.step(action)

        print()
        print(f"Reward: {reward} Done: {done}")
        print(action)
        print(show_code(code, env))


def test_wrap():
    env = karel_env.KarelEditEnv()
    task, code = env.reset()
    kagent = agent.KarelAgent(env, args)
    kagent.set_task(task)

    code, _, _, _ = env.step(Action(mutation.ADD_ACTION, ActionAddParameters(3, 'turnLeft')))
    code, _, _, _ = env.step(Action(mutation.ADD_ACTION, ActionAddParameters(3, 'turnRight')))

    kagent.select_action(code, None)

if __name__ == '__main__':
    test_wrap()
