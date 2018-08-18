import copy

import numpy as np
import torch
import torch.nn.functional as F

from program_synthesis.common.tools import saver
from program_synthesis.karel import models
from program_synthesis.karel.dataset.executor import KarelExecutor
from program_synthesis.karel.rl_agent import utils
from program_synthesis.karel.rl_agent.logger import logger_task
from program_synthesis.karel.rl_agent.utils import StepExample, ReplayBuffer, State, Task


def choice(total, sample):
    if total == 0:
        return []
    else:
        return np.random.choice(total, sample, replace=False)


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

            experience.append(StepExample(copy.deepcopy(state), action, reward, done, copy.deepcopy(new_state)))

            if done:
                success = True
                break

            state = new_state

    return success, experience


def update_step_example(trans: StepExample, new_goal, executor: KarelExecutor) -> StepExample:
    goal_code = new_goal
    new_code = trans.next_state.code

    inputs = trans.state.task.inputs.squeeze(0).data.numpy()

    outputs = np.zeros_like(inputs)

    assert inputs.shape[0] == 5

    done = True

    for ix in range(inputs.shape[0]):
        grid_inp, = np.where(inputs[ix].ravel())

        grid_out, trace = executor.execute(goal_code, None, grid_inp, record_trace=True)
        outputs[ix].ravel()[grid_out] = 1

        if done:
            _grid_out, trace = executor.execute(new_code, None, grid_inp, record_trace=True)

            if grid_out == _grid_out:
                done = False

    new_outputs = torch.from_numpy(outputs).unsqueeze(0)
    task = trans.state.task

    assert np.allclose(trans.reward, -1.)

    return StepExample(
        State(Task(task.inputs.clone(), new_outputs.clone()), trans.state.code),
        trans.action,
        trans.reward,
        done,
        State(Task(task.inputs.clone(), new_outputs.clone()), trans.next_state.code),
    )


class PolicyTrainer(object):
    def __init__(self, agent, env, args):
        self.args = args
        self.env = env
        self.vocab = env.vocab

        self.actor = agent(self.vocab, args)
        self.critic = agent(self.vocab, args)

        if not args.train_from_scratch:
            self.actor.model = models.get_model(args)

        self.critic.update_with(self.actor)

        self.criterion = F.mse_loss
        self.optimizer = torch.optim.Adam(self.actor.model.grad_parameters(), lr=args.lr)

        self.karel_executor = KarelExecutor()

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

        for ix, (s, a, r, d, ns) in enumerate(batch):
            t_code = utils.prepare_code(s.code, self.env.vocab, tensor=True)
            code_state[ix, :len(s.code)] = t_code[0]

            t_code = utils.prepare_code(ns.code, self.env.vocab, tensor=True)
            code_next_state[ix, :len(ns.code)] = t_code[0]

        # Evaluating model
        next_state_value, _ = self.critic.best_action_value(code_next_state,
                                                            utils.Task(task_next_state_I, task_next_state_O))

        # Fix this: Compute `next_state_value` only for actions that are non terminal states
        for i in range(size):
            if np.isclose(batch[i].done, 0):
                next_state_value[i] = 0.

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

        for step in range(1, self.args.num_iterations + 1):
            logger_task.info(f"Step: {step}")

            for ix in range(self.args.num_rollouts):
                logger_task.info(f"Start rollout: {ix}")
                success, experience = rollout(self.env, self.actor, self.args)
                logger_task.info(f"Success: {int(success)} Experience length: {len(experience)}")
                [replay_buffer.add(e) for e in experience]

                if self.args.her:
                    for e_ix, e in enumerate(experience):
                        future = len(experience) - e_ix - 1
                        samp = min(future, self.args.her_new_goals)

                        for c_ix in choice(future, samp):
                            new_exp = update_step_example(e, experience[-c_ix - 1].state.code, self.karel_executor)
                            replay_buffer.add(new_exp)

            self.actor.set_train(True)

            for ix in range(self.args.num_training_steps):
                logger_task.info(f"Training step: {ix}")
                batch = replay_buffer.sample(self.args.batch_size)
                loss = self.train_actor_critic(batch)
                logger_task.info(f"Loss: {loss}")

            self.actor.set_train(False)

            if (step + 1) % self.args.update_actor_it == 0:
                logger_task.info(f"Update critic with actor")
                self.critic.update_with(self.actor)

            if (step + 1) % self.args.save_actor_it == 0:
                saver.save_checkpoint(self.actor.model, self.optimizer, step, self.args.model_dir)
                saver.save_args(self.args)
