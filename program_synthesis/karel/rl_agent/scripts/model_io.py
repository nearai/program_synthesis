import torch

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis.karel.dataset import executor
from program_synthesis.karel.rl_agent.agent import KarelAgent
from program_synthesis.karel.rl_agent.environment import KarelEditEnv
from program_synthesis.tools import saver


def explore_model(model):
    from functools import reduce
    t_params = 0
    t_grad_params = 0

    for name, param in model.named_parameters():
        num_params = reduce(lambda x, y: x * y, param.size(), 1)
        print(name, num_params, param.size())
        t_params += num_params

        if param.requires_grad:
            t_grad_params += num_params

    print("Total parameters:", t_params)
    print("Total parameters that requires grad:", t_grad_params)


def load_model(model_dir, model_type, step=None):
    """
        baseline_model, baseline_eval_dataset, baseline_executor = load_model(
            '../../../data/karel/baseline-msr', 'karel-lgrl', 250100
        )
    """
    args = saver.ArgsDict(model_dir=model_dir, model_type=model_type, step=step, cuda=False, restore_map_to_cpu=True)
    saver.restore_args(args)
    arguments.backport_default_args(args)
    dataset.set_vocab(args)
    m = models.get_model(args)
    eval_dataset = dataset.get_eval_dataset(args, m)
    m.model.eval()
    the_executor = executor.get_executor(args)()
    return m, eval_dataset, the_executor


def create_model(args=None, use_trained_task_encoder=True):
    if args is None:
        args = saver.ArgsDict(
            num_iterations=100, max_rollout_length=30, replay_buffer_size=16384, max_token_per_code=75,
            num_rollouts=16, num_training_steps=16, batch_size=32, update_actor_it=10,
            rl_discount=.9, rl_eps_action=.1, rl_eps_parameter=.5, rl_alpha=.7, train_task_encoder=False,
            karel_io_enc='lgrl', lr=0.01, cuda=False, model_dir='../data')

    env = KarelEditEnv(args.max_token_per_code)
    agent = KarelAgent(env.vocab, args=args)

    if use_trained_task_encoder:
        baseline_model, _, _ = load_model(
            '../../../../data/karel/baseline-msr', 'karel-lgrl', 250100
        )

        explore_model(baseline_model.model)

        rl_sd = agent.model.state_dict()
        baseline_sd = baseline_model.model.state_dict()

        for name, params in baseline_sd.items():
            if name.startswith('encoder'):
                new_name = '.'.join(['task_encoder'] + name.split('.')[1:])
                assert new_name in rl_sd
                rl_sd[new_name] = params

        agent.model.load_state_dict(rl_sd)

    explore_model(agent.model)

    optimizer = torch.optim.Adam(agent.model.grad_parameters(), lr=args.lr)

    saver.save_checkpoint(agent.model, optimizer, 0, args.model_dir)
    saver.save_args(args)


if __name__ == '__main__':
    create_model()
