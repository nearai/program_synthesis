import torch

from program_synthesis.models.rl_agent.agent import KarelAgent
from program_synthesis.models.rl_agent.config import EPSILON, ALPHA, DISCOUNT, MAX_TOKEN_PER_CODE
from program_synthesis.models.rl_agent.environment import KarelEditEnv
from program_synthesis.models.rl_agent.logger import logger_debug, logger_task
from program_synthesis.models.rl_agent.utils import StepExample, ReplayBuffer
from program_synthesis.tools import saver


def rollout(env, agent, epsilon_greedy, max_rollout_length):
    logger_task.info("*** Start task ***")

    with torch.no_grad():
        eps = EPSILON if epsilon_greedy else None
        task, state = env.reset()
        agent.set_task(task)
        experience = []
        success = False
        for _ in range(max_rollout_length):
            action = agent.select_action(state, eps)
            new_state, reward, done, _ = env.step(action)
            assert new_state.shape[-1] <= MAX_TOKEN_PER_CODE
            experience.append(StepExample(task, state, action, reward, new_state))
            if done:
                success = True
                break
            state = new_state

        logger_task.info(f"Success: {success}")
        return success, experience


class PolicyTrainer(object):
    def __init__(self, args, agent_cls, env):
        self.args = args
        self.actor = agent_cls(env, args)
        self.critic = agent_cls(env, args)
        self.env = env

    def train_actor_critic(self, batch):
        # Train action network

        tasks = self.env.prepare_tasks([ex.task for ex in batch])
        states = self.env.prepare_states([ex.state for ex in batch])
        new_states = self.env.prepare_states([ex.new_state for ex in batch])
        actions = self.env.prepare_actions([ex.action for ex in batch])

        targets = torch.zeros(len(batch))
        values = self.critic.action_value(tasks, states)
        new_value = self.actor.best_action_value(tasks, new_states)

        for idx, ex in enumerate(batch):
            Q_s_a = values[idx][ex.action[0]]
            new_Q_s_a = 0 if ex.reward == 0 else (ex.reward + DISCOUNT * new_value[idx])
            targets[idx] = Q_s_a + ALPHA * (new_Q_s_a - Q_s_a)

        self.critic.train(tasks, states, actions, targets)

        # Train action network

    def train(self):
        replay_buffer = ReplayBuffer(self.args.replay_buffer_size, self.args.erase_factor)
        for epoch in range(self.args.num_epochs):

            for _ in range(self.args.num_episodes):
                _, experience = rollout(self.env, self.actor, True, self.args.max_rollout_length)
                replay_buffer.add(experience)

            for _ in range(self.args.num_training_steps):
                batch = replay_buffer.sample(self.args.batch_size)
                self.train_actor_critic(batch)

            if (epoch + 1) % self.args.update_actor_epoch == 0:
                self.actor.update(self.critic)
                # log info here


def main():
    args = saver.ArgsDict(
        num_epochs=100, max_rollout_length=10, replay_buffer_size=16384, erase_factor=0.01,
        num_episodes=10, num_training_steps=10, batch_size=32, update_actor_epoch=10,
        karel_io_enc='lgrl', lr=0.001, cuda=False)

    agent_cls = KarelAgent
    env = KarelEditEnv(MAX_TOKEN_PER_CODE)

    trainer = PolicyTrainer(args, agent_cls, env)
    trainer.train()


if __name__ == "__main__":
    main()
