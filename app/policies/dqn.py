from __future__ import absolute_import, division, print_function,annotations
from app.abstract.ddqn import DDQNPolicy
from app.policies.dummy_v2g import DummyV2G
from typing import Union

from app.abstract.utils import compute_avg_return
from app.policies.utils import metrics_visualization, moving_average, plot_metric
from app.models.garage_env import V2GEnvironment

import sys
import numpy as np


from typing import List
from tensorflow.keras import layers
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import trajectory, time_step


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

    def __init__(self, train_env: V2GEnvironment, eval_env: V2GEnvironment):
        # Get action tensor spec to get the number of actions (output of neural network)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        #link to the next agent, in order to simulate multi agent learning

        self._returns = []
        self._loss = []
        self._i = 0

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

    def train_reset(self):
        # Override the current implementation of the train function
        self.train_env.reset()
        self.eval_env.reset()

        self._i = 0

        print("Collect Step")

        dummy_v2g = DummyV2G(0.5)
        self.collect_data(self.train_env, dummy_v2g, self.initial_collect_steps)
        self.agent.train = common.function(self.agent.train)
        self.raw_eval_env.reset_metrics()

    def train(self):
        """
        Modiied train function to simulate multi agent learning.
        Iteratively reset the train variables for all agents. Inductive calling of single step train function (_train)
        is implemented inside the step method of the V2G Environment
        """
        self.train_reset()
        next_agent: DQNPolicy = self.train_env.get_next_agent()

        #Initialise the training constants of all the agents
        while next_agent != None:
            next_agent.train_reset()
            next_agent = self.train_env.get_next_agent()

        for _ in range(self.num_iterations):
            self._train()

    def _train(self):          #->List[time_step.TimeStep]:

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
            self.collect_data(self.train_env, self.agent.policy, 1)

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
                avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                epoch = self._i  // self.eval_interval
                total_epochs = self.num_iterations // self.eval_interval
                print(
                    "Epoch: {0}/{1} step = {2}: Average Return = {3}".format(epoch, total_epochs, step, avg_return)
                )
                self._returns.append(avg_return)

                # metrics_visualization(self.raw_eval_env.get_metrics(), (i + 1) // self.eval_interval, "dqn")
                self.raw_eval_env.hard_reset()

        except ValueError as e:
            print(self.train_env.current_time_step())
            raise e

        self.train_checkpointer.save(global_step=self.global_step.numpy())
        # plot_metric(returns, "Average Returns", 0, "plots/raw_dqn", "Average Return", no_xticks=True)
        # plot_metric(
        #     moving_average(self._loss, 240), "Training loss", 0, "plots/raw_dqn", "Loss", log_scale=True, no_xticks=True
        # )

    def collect_step(self, environment: tf_py_environment.TFPyEnvironment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        if next_time_step.is_last():
            environment.reset()
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def collect_data(self, env, policy, steps):
        for _ in range(steps):
            self.collect_step(env, policy)
