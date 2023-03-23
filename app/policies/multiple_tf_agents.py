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

import config
from config import MAX_BUFFER_SIZE
from tf_agents.trajectories import time_step as ts

from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common

import time

from app.models.tf_pwr_env import TFPowerMarketEnv

class MultiAgentPolicyWrapper(TFPolicy):
    def __init__(self,
                 policy_list: List[TFPolicy],
                 time_step_spec,
                 action_spec,
                 policy_state_spec,
                 info_spec,
                 global_step: tf.Tensor = tf.constant(0),
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
            info_spec=info_spec or tensor_spec.TensorSpec(shape=(1,), dtype=tf.int32),
        )
        self._policy_list = policy_list
        self._global_step = tf.Variable(global_step, dtype=tf.int32)
        self._collect = collect

    @property
    def wrapped_policy_list(self) -> List[TFPolicy]:
        return self._policy_list

    @tf.function
    def _action(self, time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        print('action')
        axis = 1 if tf.rank(time_step.observation) > 0 else -1
        action_tensor = tf.concat([policy.action(time_step,
                                                 policy_state,
                                                 seed).action for policy in self._policy_list],
                                  axis=axis)
        info = tf.fill([tf.shape(time_step.observation)[0], 1], self._global_step % MAX_BUFFER_SIZE)
        if self._collect:
            self._global_step.assign_add(1)
        return policy_step.PolicyStep(action=action_tensor, state=(), info=info)

class MultipleAgents:
    # num_iterations = 24 * 100 * 100
    num_iterations = config.TRAIN_ITERATIONS
    num_eval_episodes = 10  # @param {type:"integer"}
    # num_eval_episodes = 0  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset
    # eval_interval = 1  # @param {type:"integer"} # 100 days in dataset

    initial_collect_steps = 5 * 24 # @param {type:"integer"}
    # initial_collect_steps = 1 # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = MAX_BUFFER_SIZE  # @param {type:"integer"}

    batch_size = 24  # @param {type:"integer"}
    # batch_size = 1  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24 * 10  # @param {type:"integer"}

    def __init__(self, train_env: TFPowerMarketEnv,
                 eval_env: TFPowerMarketEnv,
                 agents_list: List[TFAgent],
                 ckpt_dir = "new_checkpoints"):
        self.train_env = train_env
        self.eval_env = eval_env

        self.ckpt_dir = ckpt_dir

        self.returns = tf.constant(())
        self._agent_list: List[TFAgent] = agents_list
        self._number_of_agents = len(self._agent_list)
        self._collect_data_context = data_converter.DataContext(
            time_step_spec=train_env.time_step_spec(),
            action_spec=(),
            info_spec=tensor_spec.TensorSpec(shape=(1,), dtype=tf.int32),
        )
        self.global_step = tf.Variable(0)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(self.collect_data_spec,
                                                                            batch_size=self.train_env.batch_size,
                                                                            max_length=self.replay_buffer_capacity)
        self.checkpoint = common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=3,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
        )
        if self.checkpoint.checkpoint_exists:
            print("Checkpoint found on specified folder. Continuing from there...")
        self.collect_policy = MultiAgentPolicyWrapper([agent.collect_policy for agent in self._agent_list],
                                                      self.train_env.time_step_spec(),
                                                      self.train_env.action_spec(),
                                                      (),
                                                      tensor_spec.TensorSpec((1,), tf.int32),
                                                      self.global_step.value(),
                                                      True)
        self.policy = MultiAgentPolicyWrapper([agent.policy for agent in self._agent_list],
                                                      self.eval_env.time_step_spec(),
                                                      self.eval_env.action_spec(),
                                                      (),
                                                      (),
                                                      self.global_step.value(),
                                                      False)
        steps = self.num_eval_episodes * 24
        self._metric = tf_metrics.AverageReturnMetric(batch_size=self.eval_env.batch_size,
                                                      buffer_size=steps)
        self._eval_driver = DynamicStepDriver(self.eval_env,
                                              self.policy,
                                              [self._metric],
                                              num_steps=steps)

        for agent in self._agent_list:
            agent.initialize()
            agent.train = common.function(agent.train)
            agent.private_index = self.global_step

    @property
    def number_of_agents(self):
        return self._number_of_agents

    @property
    def collect_data_spec(self):
        return self._collect_data_context.trajectory_spec

    def train(self):
        # print(self.returns[-1])
        best_avg_return = self.eval_policy()
        print(best_avg_return)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)
        iterator = iter(dataset)
        DynamicStepDriver(self.train_env,
                          self.collect_policy,
                          [lambda traj: self.replay_buffer.add_batch(traj.replace(action=())),
                           lambda traj: self.global_step.assign_add(1)],
                          num_steps=self.initial_collect_steps).run()
        collect_driver = DynamicStepDriver(self.train_env,
                                           self.collect_policy,
                                           [lambda traj: self.replay_buffer.add_batch(traj.replace(action=())),
                                            lambda traj: self.global_step.assign_add(1)],
                                           num_steps=1)
        time_step = self.train_env.reset()
        for _ in range(self.num_iterations):
            st = time.time()
            time_step, __ = collect_driver.run(time_step)
            ct = time.time()
            print('collect_time: ', ct - st)
            experience, ___ = next(iterator)
            train_loss = [agent.train(experience).loss for agent in self._agent_list]
            step = self._agent_list[0].train_step_counter
            if step % self.log_interval == 0:
                print('step = ', step, '     loss', train_loss)
            if step % self.eval_interval == 0:
                avg_return = self.eval_policy()
                epoch = tf.floor( _ / self.eval_interval)
                print('Epoch: ', epoch + 1, '            Avg_return = ', avg_return)
                self.returns.append(avg_return)
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    self.checkpoint.save(step)
                    for agent in self._agent_list:
                        agent.checkpoint_save()
            et = time.time()
            print(et - st)
        # self.checkpoint.save(global_step=self._agent_list[0].train_step_counter)
        # for agent in self._agent_list:
        #     agent.checkpoint_save()

    def eval_policy(self):
        print('Tracing eval_policy')
        self._metric.reset()
        for agent in self._agent_list:
            agent.reset_eval_steps()
        self._eval_driver.run(self.eval_env.reset())
        return self._metric.result()