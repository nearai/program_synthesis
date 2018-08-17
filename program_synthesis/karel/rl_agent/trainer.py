import copy

import torch
import torch.nn.functional as F
import numpy as np

from program_synthesis.karel.rl_agent import utils
from program_synthesis.karel.rl_agent.agent import KarelAgent
from program_synthesis.karel.rl_agent.environment import KarelEditEnv
from program_synthesis.karel.rl_agent.logger import logger_task
from program_synthesis.karel.rl_agent.utils import StepExample, ReplayBuffer
from program_synthesis.tools import saver


def rollout(env, agent, args):
    with torch.no_grad():
        state = env.reset()
        agent.set_task(state.task)

        experience = []
        success = False

        for _ in range(args.max_rollout_length):
            action = agent.select_action(state.code)
            new_state, reward, done, _ = env.step(action)

            assert len(new_state.code) <= args.max_token_per_code

            experience.append(StepExample(copy.deepcopy(state), action, reward, copy.deepcopy(new_state)))

            if done:
                success = True
                break

            state = new_state

    return success, experience


class PolicyTrainer(object):
    def __init__(self, agent, env, args):
        self.args = args
        self.env = env
        self.vocab = env.vocab

        self.actor = agent(self.vocab, args)
        self.critic = agent(self.vocab, args)
        self.critic.update(self.actor)

        self.criterion = F.mse_loss
        self.optimizer = torch.optim.Adam(self.actor.model.parameters(), lr=args.lr)

    def train_actor_critic(self, batch: "list[StepExample]"):
        size = self.args.batch_size
        alpha = self.args.rl_alpha
        discount = self.args.rl_discount

        code_pad = self.vocab.stoi(self.vocab.itos(-1))

        # Loading tensors for training
        code_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)

        task_state_I = torch.cat([ex.state.task.inputs for ex in batch])
        task_state_O = torch.cat([ex.state.task.outputs for ex in batch])

        reward = torch.Tensor([b.reward for b in batch])
        actions = [b.action for b in batch]

        code_next_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)
        task_next_state_I = torch.cat([ex.next_state.task.inputs for ex in batch])
        task_next_state_O = torch.cat([ex.next_state.task.outputs for ex in batch])

        for ix, (s, a, r, ns) in enumerate(batch):
            t_code = utils.prepare_code(s.code, self.env.vocab, tensor=True)
            code_state[ix, :len(s.code)] = t_code[0]

            t_code = utils.prepare_code(ns.code, self.env.vocab, tensor=True)
            code_next_state[ix, :len(ns.code)] = t_code[0]

        # Evaluating model
        next_state_value, _ = self.critic.best_action_value(code_next_state,
                                                            utils.Task(task_next_state_I, task_next_state_O))

        # Fix this: Compute `next_state_value` only for actions that are non terminal states
        for i in range(size):
            if np.isclose(batch[i].reward, 0):
                next_state_value = 0.

        state_value, parameter_value = self.actor.action_value_from_action(code_state,
                                                                           utils.Task(task_state_I, task_state_O),
                                                                           actions)

        target_state_value = state_value + alpha * (reward + discount * next_state_value - state_value)
        target_parameter_value = parameter_value + alpha * (reward + discount * next_state_value - parameter_value)

        loss_action = self.criterion(state_value, target_state_value)
        loss_parameter = self.criterion(parameter_value, target_parameter_value)
        loss = loss_action + loss_parameter

        assert int(torch.isnan(loss)) == 0, f"{loss} ({type(loss)})"

        # Updating model

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss)

    def train(self):
        replay_buffer = ReplayBuffer(self.args.replay_buffer_size)

        for it in range(1, self.args.num_iterations + 1):
            logger_task.info(f"Iteration: {it}")

            for ix in range(self.args.num_rollouts):
                logger_task.info(f"Start rollout: {ix}")
                success, experience = rollout(self.env, self.actor, self.args)
                logger_task.info(f"Success: {success} Experience length: {len(experience)}")
                [replay_buffer.add(e) for e in experience]

            self.actor.set_train(True)

            for ix in range(self.args.num_training_steps):
                logger_task.info(f"Training step: {ix}")
                batch = replay_buffer.sample(self.args.batch_size)
                loss = self.train_actor_critic(batch)
                logger_task.info(f"Loss: {loss}")

            self.actor.set_train(False)

            if (it + 1) % self.args.update_actor_it == 0:
                logger_task.info(f"Update critic with actor")
                self.critic.update(self.actor)


def explore_model(agent):
    from functools import reduce
    t_params = 0

    for name, param in agent.model.named_parameters():
        num_params = reduce(lambda x, y: x * y, param.size(), 1)
        print(name, num_params, param.size())
        t_params += num_params

    print("Total parameters:", t_params)


def main():
    args = saver.ArgsDict(
        num_iterations=100, max_rollout_length=30, replay_buffer_size=16384, max_token_per_code=75,
        num_rollouts=16, num_training_steps=16, batch_size=32, update_actor_it=10,
        rl_discount=.9, rl_eps_action=.1, rl_eps_parameter=.5, rl_alpha=.7,
        karel_io_enc='lgrl', lr=0.01, cuda=False)

    env = KarelEditEnv(args.max_token_per_code)

    trainer = PolicyTrainer(KarelAgent, env, args)
    trainer.train()


if __name__ == "__main__":
    main()
