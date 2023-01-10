from __future__ import absolute_import, division, print_function, annotations

import sys
from functools import reduce
from typing import List

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from app.abstract.ddqn import DDQNPolicy
# from app.abstract.utils import compute_avg_return
from app.models.garage_env import V2GEnvironment
# from config import NUMBER_OF_AGENTS
from app.policies.smart_charger import  SmartCharger

from config import GARAGE_LIST, NUMBER_OF_AGENTS


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

    def __init__(self, train_env: V2GEnvironment, eval_env: V2GEnvironment, name: str = "Default_Agent",
                 charge_list: List[np.float32] = [], model_dir=None):
        # Get action tensor spec to get the number of actions (output of neural network)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # link to the next agent, in order to simulate multi agent learning
        self._name = name
        self._returns = []
        self._loss = []
        self._i = 0
        self._policy = None
        # self.test_policy = SmartCharger(0.5) #used to check if pretrained networks word as intended
        self._next_agent = train_env.next_agent()
        self._charge_list = charge_list
        # self._eval_time_step = None
        self._eval_returns = []
        # self._episode_return_eval = None
        # self._total_return_eval = None

        # self.observation: time_step.TimeStep

        # The neural network
        # self.q_net = sequential.Sequential(
        #     [
        #         layers.Dense(units=34, activation="elu"),
        #         layers.Dense(units=128, activation="elu"),
        #         layers.BatchNormalization(),
        #         layers.Dense(
        #             num_actions,
        #             activation=None,
        #         ),
        #     ]
        # )

        if not model_dir:
            layers_list = \
                [
                    layers.Dense(units=34, activation="elu"),
                    layers.Dense(units=128, activation="elu"),
                    layers.BatchNormalization(),
                    layers.Dense(
                        num_actions,
                        activation=None,
                    ),
                ]
        else:
            keras_model = load_model(model_dir)
            layers_list = []
            i = 0
            while True:
                try:
                    layers_list.append(keras_model.get_layer(index=i))
                except IndexError:
                    print('Total number of layers in neural network: %d' % (i))
                    break
                except ValueError:
                    print('Total number of layers in neural network: %d' % (i))
                    break
                else:
                    i += 1

        self.q_net = sequential.Sequential(layers_list)

        DDQNPolicy.__init__(self, train_env, eval_env, self.q_net)

    def get_name(self):
        return self._name

    def next_agent(self):
        return self._next_agent

    # def commonize(self, function)

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
        Modified train function to simulate multi agent learning.
        Iteratively reset the train variables for all agents. Inductive calling of single step train function (_train_step)
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
        # self.raw_train_env.hard_reset()
        # self.raw_eval_env.hard_reset()
        # self.trickle_down('train_env.reset()')
        self.trickle_down('raw_train_env.hard_reset()',
                          'raw_eval_env.hard_reset()',
                          )

        self.trickle_down('train_env.reset()',
                          'eval_env.reset()',
                          'set_policy()',
                          'commonize_train()',
                          )
        # next_agent = self
        # while next_agent != None:
        #     # next_agent._policy = DummyV2G(0.5)
        #     next_agent.agent.train = common.function(next_agent.agent.train)
        #     next_agent = next_agent.next_agent()

        print("Collect Step")

        self.compute_eval_list()

        self.collect_data(self.initial_collect_steps)

        self._i = 0


        for _ in range(self.num_iterations):
            # self.trickle_down("raw_train_env.update_garage_list()")
            self.collect_data(1)  # was used in _step method
            self.trickle_down("_train_step()")
            # next_agent = selfupdate_garage_list()
            # while next_agent != None:
            #     next_agent._train_step()
            #     next_agent = next_agent.next_agent()
        self.train_checkpointer.save(global_step=self.global_step.numpy())

    def _train_step(self):  # ->List[time_step.TimeStep]:

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
                sys.stdout.flush()  # TODO denote agent's name

            if step % self.eval_interval == 0:
                # avg_return = self.compute_avg_return(self.num_eval_episodes)
                self.compute_eval_list(self.num_eval_episodes)
                epoch = self._i // self.eval_interval
                total_epochs = self.num_iterations // self.eval_interval

                print(
                    "Epoch: {0}/{1} step = {2}: Average Return (Average per day and per agent) = {3}".format(epoch,
                                                                                                             total_epochs,
                                                                                                             step,
                                                                                                             self.compute_avg_agent_loss())
                )
                # self._returns.append(avg_return)
                # self._returns.append(self.fold_eval_list)
                # metrics_visualization(self.raw_eval_env.get_metrics(), (i + 1) // self.eval_interval, "dqn")
                self.trickle_down("raw_eval_env.hard_reset()")
                # self.raw_eval_env.hard_reset()
                # next_agent = self
                # while next_agent is not None:
                #     self.raw_eval_env.hard_reset()  # TODO check if it's ok
                #     next_agent = next_agent.next_agent()

        except ValueError as e:
            print(self.train_env.current_time_step())
            raise e

        # plot_metric(returns, "Average Returns", 0, "plots/raw_dqn", "Average Return", no_xticks=True)
        # plot_metric(
        #     moving_average(self._loss, 240), "Training loss", 0, "plots/raw_dqn", "Loss", log_scale=True, no_xticks=True
        # )

    # def set_train(self):
    #     self._train_mode = True
    #
    # def set_eval(self):
    #     self._train_mode = False

    # def collect_step_eval(self):
    #     action_step = self.agent.policy.action(self._eval_time_step)
    #     self._eval_time_step = self.eval_env.step(action_step.action)
    #     self._episode_return_eval += self._eval_time_step.reward

    def fold_eval_list(self):
        return reduce(lambda x, y: x + reduce(lambda z, w: z + w, y, 0), self._eval_returns, 0) \
            / self.num_eval_episodes

    def compute_avg_agent_loss(self):
        next_agent = self
        sum = 0
        while next_agent is not None:
            ret = next_agent.fold_eval_list()
            next_agent._returns.append(ret)
            sum += ret
            next_agent = next_agent.next_agent()
        return sum / NUMBER_OF_AGENTS

    def commonize_train(self):
        self.agent.train = common.function(self.agent.train)

    def set_policy(self, policy=None):
        self._policy = self.agent.collect_policy if policy is None else policy

    def extend_eval_list(self):
        self._eval_returns.append([])

    def compute_eval_list(self, num_episodes=10):
        self.trickle_down('_eval_returns.clear()')
        for _ in range(num_episodes):
            self.trickle_down('extend_eval_list()',
                              'eval_env.reset()'
                              )
            while not self.eval_env.current_time_step().is_last():
                self._charge_list.clear()
                GARAGE_LIST.clear()
                self.collect_step(False)
        self.trickle_down('raw_eval_env.hard_reset()')

    # def compute_avg_return(self, num_episodes=10):#TODO housework
    #     # total_return = 0.0
    #     step = 0
    #     # self.trickle_down()
    #
    #     next_agent = self
    #     while next_agent != None:
    #         next_agent._total_return_eval = 0.0
    #         next_agent = next_agent.next_agent()
    #
    #     for _ in range(num_episodes):
    #
    #         next_agent = self
    #         while next_agent != None:
    #             next_agent._eval_time_step = next_agent.eval_env.reset()
    #             next_agent._episode_return_eval = 0.0
    #             next_agent = next_agent.next_agent()
    #
    #         # While the current state is not a terminal state
    #         while not self._eval_time_step.is_last():
    #             # Always clearing the action-list, that will be filled with all agents actions for the market env
    #             self._charge_list.clear()
    #             # step implemantiation for multi agent environment
    #             self.collect_step()
    #             # Increase step counter
    #             step += 1
    #
    #         # Add episode return on total_return counter
    #         next_agent = self
    #         while next_agent != None:
    #             next_agent._total_return_eval += next_agent._episode_return_eval
    #             next_agent = next_agent.next_agent()
    #
    #     # Calculate average return
    #     # avg_return = total_return / step
    #
    #     avg_agent_return = None
    #
    #     next_agent = self
    #     while next_agent != None:
    #         avg_return = next_agent._total_return_eval / step
    #         avg_agent_return += avg_return
    #         next_agent._returns.append(avg_agent_return)
    #         next_agent.raw_eval_env.hard_reset()  # TODO check if it's ok
    #         next_agent = next_agent.next_agent()
    #
    #     # Unpack value
    #     return avg_agent_return.numpy()[0] / NUMBER_OF_AGENTS

    def collect_step(self, train_mode=True):
        (env, policy) = (self.train_env, self._policy) if train_mode else (self.eval_env, self.agent.policy)
        time_step = env.current_time_step()

        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        if train_mode:
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            self.replay_buffer.add_batch(traj)
        else:
            self._eval_returns[-1].append(next_time_step.reward.numpy()[-1])  #Decide on leaving tensors

    def collect_data(self, steps):
        for _ in range(steps):
            # Always clearing the action-list, that will be filled with all agents actions for the market env.
            self._charge_list.clear()
            GARAGE_LIST.clear()
            self.collect_step()
            if self.train_env.current_time_step().is_last():
                self.trickle_down("train_env.reset()")

    def trickle_down(self, *args):  # TODO Do something about new values
        """
        trickles down methods (without arguments) to all agents under current agent. Args has to be given as strings.
        Example:
            self.trickle_down('eval_env.reset()',
                              'train_env.reset())
        """
        next_agent = self
        while next_agent != None:
            for func in args:
                action = next_agent
                for x in func.split('.'):
                    action = getattr(action, x[0:-2])() if x[-1] == ')' else getattr(action, x)
            next_agent = next_agent.next_agent()

    # def trickle_down(self, *args):  # TODO Do something about new values
    #     """
    #     trickles down methods (without arguments) to all agents under current agent. Args has to be given as strings.
    #     Example:
    #         self.trickle_down('eval_env.reset()',
    #                           'train_env.reset())
    #     """
    #     next_agent = self
    #     while next_agent != None:
    #         for func in args:
    #             action = next_agent
    #             for x in func.split('.'):
    #                 new_args =
    #                 for y in x.split(')'):
    #                     new_args.append(y.split('(')[1])
    #
    #                 action = getattr(action, x[0:-2])() if x[-1] == ')' else getattr(action, x)
    #         next_agent = next_agent.next_agent()
