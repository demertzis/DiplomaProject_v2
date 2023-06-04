import sys
from typing import List, Optional
import tensorflow as tf
from tf_agents.agents import TFAgent, data_converter
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.metrics import tf_metrics
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec

import config
from config import MAX_BUFFER_SIZE
from tf_agents.trajectories import time_step as ts, TimeStep

from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common

import time

from app.models.tf_pwr_env_2 import TFPowerMarketEnv

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
        # if len(set([policy.time_step_spec for policy in policy_list])) > 1:
        #     raise Exception('Single Agent policies have not a common time step spec')
        # if len(set([policy.action_spec for policy in policy_list])) > 1:
        #     raise Exception('Single Agent policies have not a common action step spec')
        # if len(set([policy.policy_state_spec for policy in policy_list])) > 1 or policy_list[0].policy_state_spec:
        #     raise NotImplementedError('Stateful multi-agent policies has not been implemented yet')
        # if len(set([policy.info_spec for policy in policy_list])) > 1 or policy_list[0].info_spec:
        #     raise Exception('Single Agent policies must not provide info')
        super(MultiAgentPolicyWrapper, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec=info_spec or tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64),
        )
        self._policy_list = policy_list
        self._global_step = tf.Variable(global_step, dtype=tf.int64, trainable=False, name='policy global step')
        self._collect = collect
        self._num_of_agents = len(policy_list)
        self._time_step_int_tensor = tf.Variable(0,
                                                 dtype=tf.int64,
                                                 trainable=False)
        self._agent_id_tensor = tf.range(self._num_of_agents,
                                         dtype=tf.int32) #has to be int32 because of switch_case
        self.action = tf.function(self.action, jit_compile=config.USE_JIT)
    @property
    def global_step(self):
        return self._global_step.value()
    @property
    def wrapped_policy_list(self) -> List[TFPolicy]:
        return self._policy_list

    # @tf.function(jit_compile=True)
    def _action(self, time_step: ts.TimeStep,
                      policy_state: types.NestedTensor,
                      seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        #print('Tracing _action')
        policy_action_list = [lambda: policy.action(time_step) for policy in self._policy_list]
        action_tensor = tf.map_fn(lambda id: tf.switch_case(id,
                                                            policy_action_list).action,
                                  self._agent_id_tensor,
                                  fn_output_signature=tf.float32,
                                  parallel_iterations=self._num_of_agents)[0]
        # action_tensor = tf.vectorized_map(lambda id: tf.switch_case(id,
        #                                                             policy_action_list).action,
        #                                   self._agent_id_tensor)
        # tf.print(action_tensor)
        info = tf.fill([tf.shape(time_step.observation)[0], 1], self._global_step % MAX_BUFFER_SIZE)
        # info = tf.fill([tf.shape(time_step.observation)[0]], self._global_step % MAX_BUFFER_SIZE)
        if self._collect:
            self._global_step.assign_add(1)
        return policy_step.PolicyStep(action=action_tensor, state=(), info=info)

class MultipleAgents(tf.Module):
    # num_iterations = 24 * 100 * 100
    num_iterations = config.TRAIN_ITERATIONS
    num_eval_episodes = 20  # @param {type:"integer"}
    # num_eval_episodes = 1  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset
    # episode_eval_interval = 10
    # eval_interval = 1  # @param {type:"integer"} # 100 days in dataset

    initial_collect_steps = 20 * 24 # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    # collect_episodes_per_log = 10
    epochs = 100
    replay_buffer_capacity = MAX_BUFFER_SIZE  # @param {type:"integer"}

    batch_size = 128  # @param {type:"integer"}
    # batch_size = 1  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24 * 10  # @param {type:"integer"}

    def __init__(self, train_env: TFPowerMarketEnv,
                 eval_env: TFPowerMarketEnv,
                 agents_list: List[TFAgent],
                 ckpt_dir = "new_checkpoints",
                 initial_collect_policy: Optional = None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.ckpt_dir = ckpt_dir
        self.returns = []
        self._agent_list: List[TFAgent] = agents_list
        self._number_of_agents = len(self._agent_list)
        self._collect_data_context = data_converter.DataContext(
            time_step_spec=train_env.time_step_spec(),
            action_spec=(),
            info_spec=tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64),
        )
        self._total_loss = tf.Variable([0.0] * self._number_of_agents, dtype=tf.float32, trainable=False)
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='GLOBAL_STEP_MA')
        device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) else "/device:CPU:0"
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(self.collect_data_spec,
                                                                            device=device,
                                                                            batch_size=self.train_env.batch_size,
                                                                            max_length=self.replay_buffer_capacity)
        self.checkpoint = common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=1,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
        )
        if self.checkpoint.checkpoint_exists:
            print("Checkpoint found on specified folder. Continuing from there...")
        self.collect_policy = MultiAgentPolicyWrapper([agent.collect_policy for agent in self._agent_list],
                                                       self.train_env.time_step_spec(),
                                                       self.train_env.action_spec(),
                                                       (),
                                                       tensor_spec.TensorSpec((1,), tf.int64),
                                                       self.global_step.value(),
                                                       True)
        self.policy = MultiAgentPolicyWrapper([agent.policy for agent in self._agent_list],
                                               self.eval_env.time_step_spec(),
                                               self.eval_env.action_spec(),
                                               (),
                                               (),
                                               self.global_step.value(),
                                               False)
        self._intial_policy = self.wrap_policy(initial_collect_policy, True) if initial_collect_policy else None

        steps = self.num_eval_episodes * 24
        self._metric = tf_metrics.AverageReturnMetric(batch_size=self.eval_env.batch_size,
                                                      buffer_size=steps)
        self._eval_driver = TFDriver(self.eval_env,
                                     self.policy,
                                     [self._metric],
                                     max_episodes=self.num_eval_episodes,
                                     disable_tf_function=False)
        # self._eval_driver.run = tf.function(self._eval_driver.run, jit_compile=True)
        for agent in self._agent_list:
            # agent.train = tf.function(jit_compile=config.USE_JIT)(agent.train)
            agent.private_index = self.global_step


    @property
    def number_of_agents(self):
        return self._number_of_agents

    @property
    def collect_data_spec(self):
        return self._collect_data_context.trajectory_spec

    # @tf.function
    # def _train_step(self, experience):
    #     train_list = [lambda: agent.train(experience).loss for agent in self._agent_list]
    #     train_loss = tf.map_fn(lambda id: tf.switch_case(id,
    #                                                      train_list),
    #                            tf.range(self._number_of_agents, dtype=tf.int32),
    #                            fn_output_signature=tf.float32,
    #                            parallel_iterations=self._number_of_agents)
    #     return train_loss


    def train(self):
        best_avg_return = self.eval_policy()
        print(best_avg_return)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(10)
        iterator = iter(dataset)
        if self.replay_buffer.num_frames() < self.initial_collect_steps:
            self.train_env.hard_reset()
            TFDriver(self.train_env,
                     self._intial_policy or self.collect_policy,
                     [lambda traj: self.replay_buffer.add_batch(traj.replace(action=()))],
                     max_steps=self.initial_collect_steps,
                     disable_tf_function=False).run(self.train_env.reset())

        @tf.function(jit_compile=True)
        def _train_on_experience(experience):
            train_list = [lambda: agent.train(experience).loss for agent in self._agent_list]
            train_loss = tf.map_fn(lambda id: tf.switch_case(id,
                                                             train_list),
                                   tf.range(self._number_of_agents, dtype=tf.int32),
                                   fn_output_signature=tf.float32,
                                   parallel_iterations=self._number_of_agents)
            self._total_loss.assign_add(train_loss / tf.cast(24 * self.epochs, tf.float32))

        def _train_step():
            # experience, _ = next(iterator)
            experience, _ = self.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=2)
            _train_on_experience(experience)
        collect_driver = TFDriver(self.train_env,
                                  self.collect_policy,
                                  [lambda traj: self.replay_buffer.add_batch(traj.replace(action=())),
                                   lambda traj: _train_step()],
                                  max_episodes=self.epochs,
                                  disable_tf_function=False)
        # @tf.function
        def train_epoch():
            self.train_env.hard_reset()
            collect_driver.run(self.train_env.reset())
            return self.eval_policy()

        # collect_driver.run = tf.function(collect_driver.run, jit_compile=True)
        time_step = self.train_env.reset()
        epoch_st = 0.0
        # epoch_train_st = 0.0
        loss_acc = [0.0] * self._number_of_agents


        for i in range(1, self.num_iterations+1):
            avg_return = train_epoch()
            print('Epoch: ', i, '            Avg_return = ', avg_return.numpy())
            print('Avg Train Loss: ', self._total_loss.value().numpy())
            epoch_et = time.time()
            self._total_loss.assign(self._total_loss * tf.constant(0.0, tf.float32))
            print('Epoch duration: ', epoch_et - epoch_st)
            epoch_st = epoch_et
            self.returns.append(avg_return)
            if i % 100 == 0:
                self.global_step.assign(self.collect_policy.global_step)
                self.checkpoint.save(self.global_step)
                for agent in self._agent_list:
                    agent.checkpoint_save(self.global_step)

        self.global_step.assign(self.collect_policy.global_step)
        self.checkpoint.save(self.global_step)
        for agent in self._agent_list:
            agent.checkpoint_save(self.global_step)
        # print('Final avg consumption: ', best_avg_return)
        # self.checkpoint.save(global_step=self._agent_list[0].train_step_counter)
        # for agent in self._agent_list:
        #     agent.checkpoint_save()

    def wrap_policy(self, policy_list, collect: bool):
        if not isinstance(policy_list, list):
            policy_list = [policy_list] * self._number_of_agents
        elif len(policy_list) == 1:
            policy_list = policy_list * self._number_of_agents
        elif len(policy_list) != self._number_of_agents:
            raise Exception('Policies List given should either be a single policy that will be projected to all'
                            'agents or a list with length equal to the number of agents')
        for i, policy in enumerate(policy_list):
            policy.action = self._agent_list[i].wrap_external_policy_action(policy.action, collect)
        env = self.train_env if collect else self.eval_env
        return MultiAgentPolicyWrapper(policy_list,
                                       env.time_step_spec(),
                                       env.action_spec(),
                                       (),
                                       tensor_spec.TensorSpec((1,), tf.int64),
                                       self.global_step.value(),
                                       collect)

    # @tf.function
    def eval_policy(self, policy_list: Optional[List] = None):
        #print('Tracing eval_policy')
        self._metric.reset()
        for agent in self._agent_list:
            agent.reset_eval_steps()
        self.eval_env.hard_reset()
        if policy_list is None:
            self._eval_driver.run(self.eval_env.reset())
            return self._metric.result()
        else:
            policy = self.wrap_policy(policy_list, False)
            driver = TFDriver(self.eval_env,
                              policy,
                              [self._metric],
                              max_episodes=self.num_eval_episodes,
                              disable_tf_function=False)
            # driver.run = tf.function(driver.run, jit_compile=True)
            driver.run(self.eval_env.reset())
            return self._metric.result()





