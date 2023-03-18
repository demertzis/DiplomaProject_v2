import sys
from typing import List, Optional
import tensorflow as tf
from tf_agents.agents import TFAgent, data_converter
from tf_agents.drivers import tf_driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.metrics import tf_metrics
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import reverb_utils, reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from config import MAX_BUFFER_SIZE
from tf_agents.trajectories import time_step as ts

import reverb
from tf_agents.trajectories import policy_step
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing import types
from tf_agents.utils import common

from app.models.tf_pwr_env import TFPowerMarketEnv

class MultiAgentPolicyWrapper(TFPolicy):
    def __init__(self,
                 policy_list: List[TFPolicy],
                 time_step_spec,
                 action_spec,
                 policy_state_spec,
                 info_spec,
                 collect = True
                 ):
        if len(set([policy.time_step_spec for policy in policy_list])) > 1:
            raise Exception('Single Agent policies have not a common time step spec')
        if len(set([policy.action_spec for policy in policy_list])) > 1:
            raise Exception('Single Agent policies have not a common action step spec')
        if len(set([policy.policy_state_spec for policy in policy_list])) > 1 or policy_list[0].policy_state_spec:
            raise NotImplementedError('Stateful multi-agent policies has not been implemented yet')
        if len(set([policy.info_spec for policy in policy_list])) > 1 or policy_list[0].info_spec:
            raise Exception('Single Agent policies must not provide info')

        super(MultiAgentPolicyWrapper, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec=info_spec or tensor_spec.TensorSpec(shape=(), dtype=tf.int32),
        )
        self._policy_list = policy_list
        self._global_step = tf.Variable(0, dtype=tf.int32)
        self._collect = collect

    @property
    def wrapped_policy_list(self) -> List[TFPolicy]:
        return self._policy_list

    def _action(self, time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        axis = 1 if tf.rank(time_step.observation) > 0 else -1
        action_tensor = tf.concat([policy.action(time_step,
                                                 policy_state,
                                                 seed).action for policy in self._policy_list],
                                  axis=axis)
        if self._collect:
            info = tf.fill([tf.shape(time_step.observation)[0]], self._global_step)
            self._global_step.assign_add(1)
        else:
            info = ()
        tf.print('!!!!!!!!!!')
        return policy_step.PolicyStep(action=action_tensor, state=(), info=info)


class MultipleAgents:
    num_iterations = 24 * 100 * 100
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset

    initial_collect_days = 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = MAX_BUFFER_SIZE  # @param {type:"integer"}

    batch_size = 24  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24 * 10  # @param {type:"integer"}

    def __init__(self, train_env: TFPowerMarketEnv,
                 eval_env: TFPowerMarketEnv,
                 agents_list: List[TFAgent],
                 train_dir_base = "new_checkpoints"):
        self.train_env = train_env
        self.eval_env = eval_env

        self.train_dir_base = train_dir_base

        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="action"
        # )
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(13,), dtype=np.float32, minimum=-1., maximum=1., name="observation"
        # )

        self._agent_list: List[TFAgent] = agents_list
        self._number_of_agents = len(self._agent_list)

        # time_step_spec = self.train_env.time_step_spec().step_type.shape.as_list()
        # discount_spec = self.train_env.time_step_spec().discount.shape.as_list()
        # reward_spec = self.train_env.time_step_spec().reward.shape.as_list()
        # reward_spec = reward_spec or [1]

        # time_step_spec = self.train_env.time_step_spec()._replace(
        #     step_type=tensor_spec.TensorSpec(
        #         shape=[self._number_of_agents] + time_step_spec,
        #         dtype=self.train_env.time_step_spec().step_type.dtype),
        #     discount=tensor_spec.TensorSpec(
        #         shape=[self._number_of_agents] + discount_spec,
        #         dtype=self.train_env.time_step_spec().discount.dtype),
        #     reward=tensor_spec.TensorSpec(
        #         shape=[self._number_of_agents] + reward_spec,
        #         dtype=self.train_env.time_step_spec().reward.dtype))

        # agent_action_spec = self._agent_list[0].action_spec.shape.as_list()
        # agent_action_spec = agent_action_spec or [1]
        # action_spec = tensor_spec.TensorSpec(shape=[self._number_of_agents] + agent_action_spec,
        #                                      dtype=self._agent_list[0].action_spec.dtype)

        self._collect_data_context = data_converter.DataContext(
            time_step_spec=train_env.time_step_spec(),
            action_spec=(),
            info_spec=tensor_spec.TensorSpec(shape=(), dtype=tf.int32),
        )

        # self.collect_policy = self._setup_policies([agent.collect_policy for agent in self._agent_list],
        #                                            True,
        #                                            self.train_env.batch_size)
        self.collect_policy = MultiAgentPolicyWrapper([agent.collect_policy for agent in self._agent_list],
                                                      self.train_env.time_step_spec(),
                                                      self.train_env.action_spec(),
                                                      (),
                                                      tensor_spec.TensorSpec((), tf.int32),
                                                      True)
        # self.policy = self._setup_policies([agent.policy for agent in self._agent_list],
        #                                    batch_size=self.eval_env.batch_size)
        self.policy = MultiAgentPolicyWrapper([agent.policy for agent in self._agent_list],
                                                      self.eval_env.time_step_spec(),
                                                      self.eval_env.action_spec(),
                                                      (),
                                                      (),
                                                      False)


        for agent in self._agent_list:
            agent.initialize()
            agent.train = common.function(agent.train)

    @property
    def number_of_agents(self):
        return self._number_of_agents

    @property
    def collect_data_spec(self):
        return self._collect_data_context.trajectory_spec

    def _setup_policies(self, policy_list, collect: Optional[bool] = False, batch_size: int = 1):
        """
        Something in between a wrap and a composition of the policies of the agents.
        If the program calls the policy action, every single agent action is computed
        (in the case of experience collecting the action is transformed to the load demanded
        by the agent - garage internally) and then all actions are summed into a common PolicyStep
        with the necessary info (global_step) attached so that agents can individually derive
        the garage state from their respectable TensorArrays.
        In every other call, the policy acts as a pass-through wrapper to the 1st agent policy
        """
        class MultipleAgentsPolicy:
            def __init__(self, policy_list: List[TFPolicy], collect: bool = collect, batch_size: int = 1):
                self._policy_list = policy_list
                self.global_step = tf.Variable(tf.zeros([batch_size], tf.int32),
                                               shape=[batch_size],
                                               dtype=tf.int32) if collect else None
                self._batch_size = batch_size

            def __getattr__(self, item):
                if item=='action':
                    return self._multi_agent_action
                else:
                    return getattr(self._policy_list[0], item)

            def _multi_agent_action(self,
                       time_step,
                       policy_state=(),
                       seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
                attr_list = [policy.action(time_step, policy_state, seed) for policy in self._policy_list]
                if all([isinstance(attr, PolicyStep) for attr in attr_list]):
                    kwargs = {'action': tf.concat([step.action for step in attr_list], 0)}
                    if self.global_step is not None:
                        kwargs['info'] = self.global_step
                        if not time_step.is_last():
                            self.global_step.assign_add(tf.reshape(1,
                                                                   self.global_step.get_shape()))
                    return attr_list[0].replace(**kwargs)
        return MultipleAgentsPolicy(policy_list, collect, batch_size)

    def train(self):

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # my_observer = lambda traj: tf.cond(
        #     tf.less_equal(traj.step_type, 1),
        #     lambda: replay_buffer.add_batch(traj.replace(action=(),
        #                                                  policy_info=tf.expand_dims(traj.policy_info, 0),)),
        #     lambda: tf.no_op())

        time_step, _ = tf_driver.TFDriver(self.train_env,
                                          self.collect_policy,
                                          [lambda traj: replay_buffer.add_batch(traj.replace(action=()))],
                                          max_episodes=self.initial_collect_days).run(self.train_env.reset())
        # DynamicStepDriver(
        #     self.train_env,
        #     self.collect_policy,
        #     [my_observer],
        #     num_steps=self.initial_collect_steps).run()

        # DynamicStepDriver(
        #     self.train_env,
        #     self.collect_policy,
        #     [my_observer],
        #     num_steps=1)
        collect_driver = tf_driver.TFDriver(
            self.train_env,
            self.collect_policy,
            # [my_observer],
            [lambda traj: replay_buffer.add_batch(traj.replace(action=()))],
            # disable_tf_function=True,
            max_steps=1)

        # time_step = self.train_env.reset()

        for _ in range(self.num_iterations):
            time_step, __ = collect_driver.run(time_step)

            experience, ___ = next(iterator)
            train_loss = [agent.train(experience).loss for agent in self._agent_list]

            if _ % self.log_interval == 0:  # TODO make it show avg agent loss if possible
                remainder = _ % self.eval_interval
                percentage = ((remainder if remainder != 0 else self.eval_interval) * 70) // self.eval_interval
                sys.stdout.write("\r")
                sys.stdout.write("\033[K")
                sys.stdout.write(
                    f'[{"=" * percentage + " " * (70 - percentage)}] {self._name} self._loss: {train_loss} ')
                sys.stdout.flush()  # TODO denote agent's name

            if _ % self.eval_interval == 0:  # TODO fix evaluation, it's executed by each agent (second agent fails)
                total_return = self.eval_policy()
                epoch = self._i // self.eval_interval
                total_epochs = self.num_iterations // self.eval_interval
                print(
                    "Epoch: {0}/{1} step = {2}: Average"
                    " Return (Average per day and per agent) = {3}".format(epoch,
                                                                           total_epochs,
                                                                           _,
                                                                           total_return / len(self._agent_list))
                )

            for agent in self._agent_list:
                agent.checkpointer.save(self.global_step.numpy())

    def eval_policy(self, policy: Optional = None):
        eval_policy = policy or self.policy
        for agent in self._agent_list:
            agent.reset_eval_steps()
        episodes = self.num_eval_episodes * 24
        metric = tf_metrics.AverageReturnMetric(
            batch_size=self.eval_env.batch_size,
            buffer_size=episodes
        )
        tf_driver.TFDriver(
            self.eval_env,
            eval_policy,
            [metric],
            # disable_tf_function=True,
            max_steps=episodes).run(self.eval_env.reset())
        return metric.result()