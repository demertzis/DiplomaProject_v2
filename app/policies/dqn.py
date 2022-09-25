from __future__ import absolute_import, division, print_function, annotations

import sys
from typing import List

import numpy as np
from tensorflow.keras import layers
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from app.abstract.ddqn import DDQNPolicy
# from app.abstract.utils import compute_avg_return
from app.models.garage_env import V2GEnvironment
from app.policies.dummy_v2g import DummyV2G
from config import NUMBER_OF_AGENTS


class DQNPolicy(DDQNPolicy):
    num_iterations = 24 * 30 * 200  # @param {type:"integer"}

    initial_collect_steps = 24 * 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 1000000  # @param {type:"integer"}

    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-4  # @param {type:"number"}
    log_interval = 24  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 30 * 8  # @param {type:"integer"}

    train_dir = "checkpoints"

    def __init__(self, train_env: V2GEnvironment, eval_env: V2GEnvironment, charge_list: List[np.float32] = []):
        # Get action tensor spec to get the number of actions (output of neural network)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # link to the next agent, in order to simulate multi agent learning

        self._returns = []
        self._loss = []
        self._i = 0
        self._policy = None
        self._next_agent = train_env.next_agent()
        self._charge_list = charge_list
        self._eval_time_step = None
        self._episode_return_eval = None
        self._total_return_eval = None

        # The neural network
        self.q_net = sequential.Sequential(
            [
                layers.Dense(units=34, activation="elu"),
                layers.Dense(units=128, activation="elu"),
                layers.BatchNormalization(),
                layers.Dense(
                    num_actions,
                    activation=None,
                ),
            ]
        )

        DDQNPolicy.__init__(self, train_env, eval_env, self.q_net)

    def next_agent(self):
        return self._next_agent

    # def train_reset(self):
    #     # Override the current implementation of the train function
    #
    #
    #     print("Collect Step")
    #
    #     # dummy_v2g = DummyV2G(0.5)
    #     self.collect_data(self.initial_collect_steps)
    #     self.agent.train = common.function(self.agent.train)
    #
    #     self.raw_eval_env.reset_metrics()

    def train(self):
        """
        Modiied train function to simulate multi agent learning.
        Iteratively reset the train variables for all agents. Inductive calling of single step train function (_train)
        is implemented inside the step method of the V2G Environment
        """
        # Initialise the training constants of all the agents
        # self.train_env.reset()
        # self.eval_env.reset()
        # self._policy = DummyV2G(0.5)
        # self.agent.train = common.function(self.agent.train)
        # self.raw_eval_env.reset_metrics()
        #
        # next_agent: DQNPolicy = self._next_agent
        next_agent = self
        while next_agent != None:
            next_agent.train_env.reset()
            next_agent.eval_env.reset()
            next_agent._policy = DummyV2G(0.5)
            next_agent.agent.train = common.function(next_agent.agent.train)
            next_agent.raw_eval_env.reset_metrics()
            next_agent = next_agent.next_agent()

        print("Collect Step")

        self.collect_data(self.initial_collect_steps)

        self._i = 0

        for _ in range(self.num_iterations):
            self.collect_data(1)  # was used in _step method
            self._train()

    def _train(self):  # ->List[time_step.TimeStep]:

        # # Override the current implementation of the train function
        # self.train_env.reset()
        # self.eval_env.reset()
        #
        #
        #
        #
        # print("Collect Step")
        #
        # dummy_v2g = DummyV2G(0.5)
        # self.collect_data(self.train_env, dummy_v2g, self.initial_collect_steps)
        # self.agent.train = common.function(self.agent.train)
        # self.raw_eval_env.reset_metrics()
        # returns = []
        # loss = []

        try:
            self._i += 1

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            self._loss.append(train_loss)

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                remainder = step % self.eval_interval
                percentage = ((remainder if remainder != 0 else self.eval_interval) * 70) // self.eval_interval
                sys.stdout.write("\r")
                sys.stdout.write("\033[K")
                sys.stdout.write(f'[{"=" * percentage + " " * (70 - percentage)}] self._loss: {train_loss} ')
                sys.stdout.flush()

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.num_eval_episodes)
                epoch = self._i // self.eval_interval
                total_epochs = self.num_iterations // self.eval_interval
                print(
                    "Epoch: {0}/{1} step = {2}: Average Return (Average per day and per agent) = {3}".format(epoch,
                                                                                                             total_epochs,
                                                                                                             step,
                                                                                                             avg_return)
                )
                self._returns.append(avg_return)
                # metrics_visualization(self.raw_eval_env.get_metrics(), (i + 1) // self.eval_interval, "dqn")
                next_agent = self
                while next_agent is not None:
                    self.raw_eval_env.hard_reset()
                    next_agent = next_agent.next_agent()

        except ValueError as e:
            print(self.train_env.current_time_step())
            raise e

        self.train_checkpointer.save(global_step=self.global_step.numpy())
        # plot_metric(returns, "Average Returns", 0, "plots/raw_dqn", "Average Return", no_xticks=True)
        # plot_metric(
        #     moving_average(self._loss, 240), "Training loss", 0, "plots/raw_dqn", "Loss", log_scale=True, no_xticks=True
        # )

    def collect_step_eval(self):
        action_step = self.agent.policy.action(self._eval_time_step)
        self._eval_time_step = self.eval_env.step(action_step.action)
        self._episode_return_eval += self._eval_time_step.reward

    def compute_avg_return(self, num_episodes=10):
        total_return = 0.0
        step = 0

        next_agent = self
        while next_agent != None:
            next_agent._total_return_eval = 0.0
            next_agent = next_agent.next_agent()

        for _ in range(num_episodes):

            next_agent = self
            while next_agent != None:
                next_agent._eval_time_step = next_agent.eval_env.reset()
                next_agent._episode_return_eval = 0.0
                next_agent = next_agent.next_agent()

            # While the current state is not a terminal state
            while not self._eval_time_step.is_last():
                # Always clearing the action-list, that will be filled with all agents actions for the market env
                self._charge_list.clear()
                # step implemantiation for multi agent environment
                self.collect_step_eval()
                # Increase step counter
                step += 1

            # Add episode return on total_return counter
            next_agent = self
            while next_agent != None:
                next_agent._total_return_eval += next_agent._episode_return_eval
                next_agent = next_agent.next_agent()

        # Calculate average return
        # avg_return = total_return / step

        avg_agent_return = None

        next_agent = self
        while next_agent != None:
            avg_return = next_agent._total_return_eval / step
            avg_agent_return += avg_return
            next_agent._returns.append(avg_agent_return)
            next_agent.raw_eval_env.hard_reset()
            next_agent = next_agent.next_agent()

        # Unpack value
        return avg_agent_return.numpy()[0] / NUMBER_OF_AGENTS

    def collect_step(self):
        time_step = self.train_env.current_time_step()
        action_step = self._policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        if next_time_step.is_last():
            self.train_env.reset()
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def collect_data(self, steps):
        for _ in range(steps):
            # Always clearing the action-list, that will be filled with all agents actions for the market env.
            self._charge_list.clear()
            self.collect_step()
