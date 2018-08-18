from program_synthesis.common.tools import saver
from program_synthesis.karel.rl_agent.agent import KarelAgent
from program_synthesis.karel.rl_agent.environment import KarelEditEnv
from program_synthesis.karel.rl_agent.trainer import PolicyTrainer


def shortcut():
    args = saver.ArgsDict(
        num_iterations=100, max_rollout_length=30, replay_buffer_size=16384, max_token_per_code=75,
        num_rollouts=16, num_training_steps=16, batch_size=32,
        update_actor_it=10, save_actor_it=10, lr=0.01,
        rl_discount=.9, rl_eps_action=.1, rl_eps_parameter=.5, rl_alpha=.7, her=True, her_new_goals=30,
        karel_io_enc='lgrl', cuda=False, train_task_encoder=False, train_from_scratch=True)

    env = KarelEditEnv(args.max_token_per_code)

    trainer = PolicyTrainer(KarelAgent, env, args)
    trainer.train()


if __name__ == "__main__":
    shortcut()
