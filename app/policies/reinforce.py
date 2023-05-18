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
from app.abstract.utils import MyCheckpointer
from app.models.garage_env import V2GEnvironment
from config import GARAGE_LIST, NUMBER_OF_AGENTS


# from config import NUMBER_OF_AGENTS


class DQNPolicy(DDQNPolicy):
    num_iterations = 24 * 100 * 100
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset

    initial_collect_steps = 24 * 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 1000000  # @param {type:"integer"}

    batch_size = 32  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24 * 10  # @param {type:"integer"}

    train_dir_base = "checkpoints"

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
        # self.test_policy = SmartCharger(0.5) #used to check if pretrained networks work as intended
        self._next_agent = train_env.next_agent()
        self._charge_list = charge_list
        # self._eval_time_step = None
        self._eval_returns = []
        self._last_eval = None
        self._best_eval = -np.inf

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
                    print('{0}: Total number of layers in neural network: {1}'.format(self._name, i))
                    break
                except ValueError:
                    print('{0}: Total number of layers in neural network: {1}'.format(self._name, i))
                    break
                else:
                    i += 1

        self.q_net = sequential.Sequential(layers_list)

        self.train_dir = self.train_dir_base + '/' + \
                         str(NUMBER_OF_AGENTS) + \
                         '_agents/' + \
                         train_env.get_reward_func_name() + '/' + \
                         self._name

        DDQNPolicy.__init__(self, train_env, eval_env, self.q_net, self._name)
        if not self.train_checkpointer.checkpoint_exists:
            self.train_checkpointer.save(global_step=self.global_step)
        self.train_checkpointer = MyCheckpointer(
            ckpt_dir=self.train_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.global_step,
        )
        self.final_checkpointer = MyCheckpointer(
            ckpt_dir=self.train_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.global_step,
            replay_buffer=self.replay_buffer,
        )

    def _train_dir_update(self, update_buffer = True):
        self.train_dir = self.train_dir_base + '/' + \
                         str(NUMBER_OF_AGENTS) + \
                         '_agents/' + \
                         self.raw_train_env.get_reward_func_name() + '/' + \
                         self._name

        checkpointer_kwargs = {'ckpt_dir': self.train_dir,
                              'max_to_keep': 1,
                              'agent': self.agent,
                              'policy': self.agent.policy,
                              'global_step': self.global_step}

        if update_buffer: checkpointer_kwargs['replay_buffer'] = self.replay_buffer

        self.train_checkpointer = common.Checkpointer(**checkpointer_kwargs)

        # If a checkpoint already exists in the "train_dir" folder, output a message informing the user
        if self.train_checkpointer.checkpoint_exists:
            if update_buffer:
                print("Checkpoint found. Continuing from checkpoint...")
            else:
                print('Restoring best performing policy from checkpoint...')
        else:
            print("New Train configuration. Creating new checkpoint folder...")

        # This will restore the agent, policy and replay buffer from the checkpoint, if it exists
        # Or it will just initialize everything to their initial setup
        self.train_checkpointer.initialize_or_restore()
        # Create a dataset and an iterator instance, this helps to create batches of data in training
        if not self.train_checkpointer.checkpoint_exists: self.train_checkpointer.save(global_step=self.global_step)
        if update_buffer:
            dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
            ).prefetch(3)
            self.iterator = iter(dataset)

    def train(self):
        """
        Modified train function to simulate multi agent learning.
        Iteratively reset the train variables for all agents. Inductive calling of every single step (env.step)
        is implemented inside the step method of the V2G Environment (by calling the collect_step function of the
        next agent
        """
        if self.train_dir_base + '/' + \
                str(NUMBER_OF_AGENTS) + \
                '_agents/' + \
                self.raw_train_env.get_reward_func_name() + '/' + \
                self._name != self.train_dir:
            self.trickle_down('_train_dir_update()')

        self.trickle_down('raw_train_env.hard_reset()',
                          'raw_eval_env.hard_reset()',
                          )
        self.trickle_down('train_env.reset()',
                          'eval_env.reset()',
                          'set_policy()',
                          'commonize_train()',
                          )
        self._best_eval = self.evaluate_policy()
        print("Average agent loss before training: {0}".format(self._best_eval))
        # next_agent = self
        # while next_agent != None:
        #     # next_agent._policy = DummyV2G(0.5)
        #     next_agent.agent.train = common.function(next_agent.agent.train)
        #     next_agent = next_agent.next_agent()

        print("Collect Step")

        # self.compute_eval_list

        self.collect_data(self.initial_collect_steps) if self.replay_buffer.num_frames().numpy() == 0 else None

        self._i = 0

        for _ in range(self.num_iterations):
            self.collect_data(1)  # was used in _step method
            self.trickle_down("_train_step()")

            step = self.agent.train_step_counter.numpy()#TODO check this too
            if step % self.eval_interval == 0:  # TODO fix evaluation, it's executed by each agent (second agent fails)
                self.compute_eval_list(self.num_eval_episodes)
                epoch = self._i // self.eval_interval
                total_epochs = self.num_iterations // self.eval_interval

                print(
                    "Epoch: {0}/{1} step = {2}: Average Return (Average per day and per agent) = {3}".format(epoch,
                                                                                                             total_epochs,
                                                                                                             step,
                                                                                                             self.compute_avg_agent_loss())
                )
                # self.trickle_down('_update_best_eval()')#TODO decide on saving replay buffer seperatelly
                if self._last_eval > self._best_eval:
                    self._best_eval = self._last_eval
                    self.train_checkpointer.save(global_step=self.global_step.numpy())
        self.train_checkpointer.initialize_or_restore().expect_partial()
        self.final_checkpointer.save(global_step=self.global_step.numpy())

    def _train_step(self):  # ->List[time_step.TimeStep]:

        try:
            self._i += 1  # TODO check if i is redundant

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            self._loss.append(train_loss)

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:  # TODO make it show avg agent loss if possible
                remainder = step % self.eval_interval
                percentage = ((remainder if remainder != 0 else self.eval_interval) * 70) // self.eval_interval
                sys.stdout.write("\r")
                sys.stdout.write("\033[K")
                sys.stdout.write(
                    f'[{"=" * percentage + " " * (70 - percentage)}] {self._name} self._loss: {train_loss} ')
                sys.stdout.flush()  # TODO denote agent's _name



        except ValueError as e:
            print(self.train_env.current_time_step())
            raise e


    def _fold_eval_list(self):
        self._last_eval = reduce(lambda x, y: x + reduce(lambda z, w: z + w, y, 0), self._eval_returns, 0) / \
                          len(self._eval_returns) if len(self._eval_returns) > 0 else 0.0
        return self._last_eval

    def compute_avg_agent_loss(self):
        next_agent = self
        sum = 0
        while next_agent is not None:
            ret = next_agent._fold_eval_list()
            next_agent._returns.append(ret)
            sum += ret
            next_agent = next_agent.next_agent()
        return sum / NUMBER_OF_AGENTS

    def commonize_train(self):
        self.agent.train = common.function(self.agent.train)

    def set_policy(self, policy=None):
        self._policy = policy

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

    def collect_step(self, train_mode=True):
        (env, policy) = (self.train_env, self.agent.collect_policy) if train_mode else (
                        self.eval_env, self.agent.policy if self._policy is None else self._policy)
        time_step = env.current_time_step()

        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        if train_mode:
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            self.replay_buffer.add_batch(traj)
        else:
            self._eval_returns[-1].append(next_time_step.reward.numpy()[-1])  # Decide on leaving tensors

    def collect_data(self, steps: object) -> object:
        for _ in range(steps):
            # Always clearing the action-list, that will be filled with all agents actions for the market env.
            self._charge_list.clear()
            GARAGE_LIST.clear()
            self.collect_step()
            if self.train_env.current_time_step().is_last():
                self.trickle_down("train_env.reset()")

    def evaluate_policy(self, policy=None):  # TODO check if it works
        next_agent = self
        while next_agent != None:
            next_agent.set_policy(policy)
            next_agent = next_agent.next_agent()
        old_last_eval = self._last_eval
        self.trickle_down('raw_eval_env.hard_reset()')
        self.compute_eval_list(num_episodes=self.num_eval_episodes)
        avg_loss = self.compute_avg_agent_loss()
        self._last_eval = old_last_eval
        self.trickle_down('set_policy()')
        return avg_loss

    def get_name(self):
        return self._name

    def next_agent(self):
        return self._next_agent

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
