import sys
from typing import List, Optional
import tensorflow as tf
from tf_agents.agents import TFAgent, data_converter
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
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

from app.models.tf_pwr_env import TFPowerMarketEnv

# class MultiAgentPolicyWrapper(TFPolicy):
#     def __init__(self,
#                  policy_list: List[TFPolicy],
#                  time_step_spec,
#                  action_spec,
#                  policy_state_spec,
#                  info_spec,
#                  global_step: tf.Tensor = tf.constant(0),
#                  collect = True
#                  ):
#         # if len(set([policy.time_step_spec for policy in policy_list])) > 1:
#         #     raise Exception('Single Agent policies have not a common time step spec')
#         # if len(set([policy.action_spec for policy in policy_list])) > 1:
#         #     raise Exception('Single Agent policies have not a common action step spec')
#         # if len(set([policy.policy_state_spec for policy in policy_list])) > 1 or policy_list[0].policy_state_spec:
#         #     raise NotImplementedError('Stateful multi-agent policies has not been implemented yet')
#         # if len(set([policy.info_spec for policy in policy_list])) > 1 or policy_list[0].info_spec:
#         #     raise Exception('Single Agent policies must not provide info')
#         super(MultiAgentPolicyWrapper, self).__init__(
#             time_step_spec,
#             action_spec,
#             policy_state_spec,
#             info_spec=info_spec or tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64),
#         )
#         self._policy_list = policy_list
#         self._global_step = tf.Variable(global_step, dtype=tf.int64, trainable=False)
#         self._collect = collect
#         self._num_of_agents = len(policy_list)
#         # self._time_step_float_16_tensors = tf.Variable(tf.zeros((14,), tf.float16), trainable=False)
#         self._time_step_float_32_tensors = tf.Variable(tf.zeros((13 + self._num_of_agents + 1,),
#                                                                 tf.float32),
#                                                        trainable=False)
#         # self._time_step_float_32_tensors = tf.Variable(tf.zeros((13 + self._num_of_agents + 1,), tf.float32), trainable=False)
#         self._time_step_int_tensor = tf.Variable(0,
#                                                  dtype=tf.int64,
#                                                  trainable=False)
#         self._agent_id_tensor = tf.range(self._num_of_agents,
#                                          dtype=tf.int32) #has to be int32 because of switch_case
#         self._policy_action_dict = [lambda: policy.action(self._get_time_step(),
#                                                           ()) for policy in self._policy_list]
#     @property
#     def global_step(self):
#         return self._global_step.value()
#
#     def _get_time_step(self):
#         float_32_members = self._time_step_float_32_tensors
#         int_members = self._time_step_int_tensor
#         observation = tf.expand_dims(float_32_members[:13], axis=0)
#         # discount = tf.expand_dims(float_16_members[13], axis=0)
#         discount = tf.expand_dims(float_32_members[0], axis=0)
#         # reward = tf.expand_dims(float_32_members, axis=0)
#         reward = tf.expand_dims(float_32_members[1:], axis=0)
#         step_type = tf.expand_dims(int_members, axis=0)
#         return TimeStep(observation=observation,
#                         reward=reward,
#                         discount=discount,
#                         step_type=step_type)
#
#     @property
#     def wrapped_policy_list(self) -> List[TFPolicy]:
#         return self._policy_list
#
#     @tf.function
#     def _action(self, time_step: ts.TimeStep,
#                       policy_state: types.NestedTensor,
#                       seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
#         #print('Tracing _action')
#         # float_time_16_step_members = tf.concat((time_step.observation[0],
#         #                                         time_step.discount), axis=0)
#         # float_time_32_step_members = time_step.reward[0]
#         float_time_32_step_members = tf.concat((time_step.observation[0],
#                                                 time_step.discount,
#                                                 time_step.reward[0]), axis=0)
#         int_time_step_members = time_step.step_type[0]
#         self._time_step_float_32_tensors.assign(float_time_32_step_members)
#         self._time_step_int_tensor.assign(tf.cast(int_time_step_members, tf.int64))
#         action_tensor = tf.map_fn(lambda id: tf.switch_case(id,
#                                                             self._policy_action_dict).action,
#                                   self._agent_id_tensor,
#                                   fn_output_signature=tf.float32,
#                                   parallel_iterations=self._num_of_agents)[0]
#         info = tf.fill([tf.shape(time_step.observation)[0], 1], self._global_step % MAX_BUFFER_SIZE)
#         # info = tf.fill([tf.shape(time_step.observation)[0]], self._global_step % MAX_BUFFER_SIZE)
#         if self._collect:
#             self._global_step.assign_add(1)
#         return policy_step.PolicyStep(action=action_tensor, state=(), info=info)
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
        self._global_step = tf.Variable(global_step, dtype=tf.int64, trainable=False)
        self._collect = collect
        self._num_of_agents = len(policy_list)
        self._time_step_int_tensor = tf.Variable(0,
                                                 dtype=tf.int64,
                                                 trainable=False)
        self._agent_id_tensor = tf.range(self._num_of_agents,
                                         dtype=tf.int32) #has to be int32 because of switch_case
    @property
    def global_step(self):
        return self._global_step.value()
    @property
    def wrapped_policy_list(self) -> List[TFPolicy]:
        return self._policy_list

    @tf.function
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
    num_eval_episodes = 10  # @param {type:"integer"}
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
                 ckpt_dir = "new_checkpoints"):
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
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(self.collect_data_spec,
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
        steps = self.num_eval_episodes * 24
        self._metric = tf_metrics.AverageReturnMetric(batch_size=self.eval_env.batch_size,
                                                      buffer_size=steps)
        self._eval_driver = DynamicEpisodeDriver(self.eval_env,
                                                 self.policy,
                                                 [self._metric],
                                                 num_episodes=self.num_eval_episodes)
        for agent in self._agent_list:
            agent.train = tf.function(jit_compile=True)(agent.train)
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
        # tf.config.run_functions_eagerly(True)
        best_avg_return = self.eval_policy()
        print(best_avg_return)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(10)
        iterator = iter(dataset)
        # tf.config.run_functions_eagerly(True)
        if self.replay_buffer.num_frames() < self.initial_collect_steps:
            DynamicStepDriver(self.train_env,
                              self.collect_policy,
                              [lambda traj: self.replay_buffer.add_batch(traj.replace(action=()))],
                              num_steps=self.initial_collect_steps).run()

        # train_list = [lambda exp: agent.train(exp) for agent in self._agent_list]
        @tf.function()
        def _train_step():
            experience, _ = next(iterator)
            train_list = [lambda: agent.train(experience).loss for agent in self._agent_list]
            train_loss = tf.map_fn(lambda id: tf.switch_case(id,
                                                             train_list),
                                   tf.range(self._number_of_agents, dtype=tf.int32),
                                   fn_output_signature=tf.float32,
                                   parallel_iterations=self._number_of_agents)
            # train_loss = tf.vectorized_map(lambda id: tf.switch_case(id,
            #                                                          train_list),
            #                                tf.range(self._number_of_agents, dtype=tf.int32))
            # if tf.reduce_any(tf.math.is_nan(train_loss)):
            #     tf.print('Train_loss is nan at step: ', self._agent_list[0].train_step_counter)
            self._total_loss.assign_add(train_loss / tf.cast(24 * self.epochs, tf.float32))
            # return train_loss

        # collect_driver = DynamicStepDriver(self.train_env,
        #                                    self.collect_policy,
        #                                    [lambda traj: self.replay_buffer.add_batch(traj.replace(action=())),
        #                                     lambda traj: _train_step()],
        #                                    num_steps=self.collect_steps_per_iteration)
        collect_driver = DynamicEpisodeDriver(self.train_env,
                                              self.collect_policy,
                                              [lambda traj: self.replay_buffer.add_batch(traj.replace(action=())),
                                               lambda traj: _train_step()],
                                              num_episodes=self.epochs)
        time_step = self.train_env.reset()
        epoch_st = time.time()
        # epoch_train_st = 0.0
        loss_acc = [0.0] * self._number_of_agents

        for i in range(1, self.num_iterations+1):
            time_step, __ = collect_driver.run(time_step)
            # print('collect_time: ', time.time() - st)
            # experience, ___ = next(iterator)
            # tst = time.time()
            # train_loss = [agent.train(experience).loss for agent in self._agent_list]
            # train_loss = self._train_step(experience)
            # loss_acc = [loss_acc[e] + t.numpy() for e, t in enumerate(train_loss)]
            # epoch_train_st += (time.time() - tst)
            # if step % self.log_interval == 0:
            #     # average_loss = [l / self.log_interval for l in loss_acc]
            #     average_loss = [l / self.log_interval for l in loss_acc]
            #     average_loss = sum(average_loss) / len(average_loss)
            #     # print('step = ', step, '     loss', train_loss)
            #     print('step = ', step.numpy(), '     loss = ', average_loss)
            #     loss_acc = [0.0] * self._number_of_agents
            avg_return = self.eval_policy()
            print('Epoch: ', i, '            Avg_return = ', avg_return)
            print('Avg Train Loss: ', self._total_loss.value().numpy())
            epoch_et = time.time()
            self._total_loss.assign(self._total_loss * tf.constant(0.0, tf.float32))
            print('Epoch duration: ', epoch_et - epoch_st)
            epoch_st = epoch_et
            self.returns.append(avg_return)

            # if avg_return > best_avg_return:
            #     self.global_step.assign(self.collect_policy.global_step)
            #     self.checkpoint.save(self.global_step)
            #     for agent in self._agent_list:
            #         agent.checkpoint_save(self.global_step)

            # if i % self.episode_eval_interval == 0:
            #     avg_return = self.eval_policy()
            #     epoch = i // self.episode_eval_interval
            #     print('Epoch: ', epoch, '            Avg_return = ', avg_return)
            #     epoch_et = time.time()
            #     print('Epoch duration: ', epoch_et - epoch_st)
            #     epoch_st = epoch_et
            #     self.returns.append(avg_return)

            # if step % self.eval_interval == 0:
            #     avg_return = self.eval_policy()
            #     epoch = int(_ / self.eval_interval)
            #     print('Epoch: ', epoch + 1, '            Avg_return = ', avg_return)
            #     epoch_et = time.time()
            #     print('Epoch duration: ', epoch_et - epoch_st)
            #     #print('Train duration: ', epoch_train_st)
            #     epoch_train_st = 0.0
            #     epoch_st = epoch_et
            #     self.returns.append(avg_return)

                # if avg_return > best_avg_return:
                #     best_avg_return = avg_return
                #     self.global_step.assign(self.collect_policy.global_step)
                #     self.checkpoint.save(self.global_step)
                #     for agent in self._agent_list:
                #         agent.checkpoint_save(self.global_step)
        self.global_step.assign(self.collect_policy.global_step)
        self.checkpoint.save(self.global_step)
        for agent in self._agent_list:
            agent.checkpoint_save(self.global_step)

        # print('Final avg consumption: ', best_avg_return)
        # self.checkpoint.save(global_step=self._agent_list[0].train_step_counter)
        # for agent in self._agent_list:
        #     agent.checkpoint_save()

    @tf.function
    def train_single_agent(self, experience, index: int):
        return self._agent_list[index].train(experience).loss

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
            if len(policy_list) == 1:
                policy_list = policy_list * self._number_of_agents
            elif len(policy_list) != self._number_of_agents:
                raise Exception('Policies List given should either be a single policy that will be projected to all'
                                'agents or a list with length equal to the number of agents')
            # policy_list = [agent.wrap_external_policy_action(policy_list[i].action) \
            #                for i, agent in enumerate(self._agent_list)]
            for i, policy in enumerate(policy_list):
                policy.action = self._agent_list[i].wrap_external_policy_action(policy.action)
            policy = MultiAgentPolicyWrapper(policy_list,
                                             self.eval_env.time_step_spec(),
                                             self.eval_env.action_spec(),
                                             (),
                                             (),
                                             self.global_step.value(),
                                             False)
            driver = DynamicEpisodeDriver(self.eval_env,
                                          policy,
                                          [self._metric],
                                          num_episodes=self.num_eval_episodes)
            driver.run(self.eval_env.reset())
            return self._metric.result()





