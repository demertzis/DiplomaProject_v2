import sys
from functools import reduce, partial
from typing import List, Optional
import tensorflow as tf
from keras.layers import Activation
from tensorflow import DType
from tf_agents.agents import TFAgent, data_converter
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import Sequential, Network
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import value_ops

import config
from app.abstract.multi_dqn import MultiDdqnAgent
from app.abstract.utils import MyNetwork, my_discounted_return, my_index_with_actions, my_to_n_step_transition, \
    MyTFDriver
from config import MAX_BUFFER_SIZE
from tf_agents.trajectories import time_step as ts, TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common
import tf_agents
import time

from app.models.tf_pwr_env_2 import TFPowerMarketEnv

class MultiAgentSingleModelPolicy(TFPolicy):
    def __init__(self,
                 policy,
                 agent_list,
                 time_step_spec,
                 action_spec,
                 policy_state_spec,
                 info_spec,
                 last_action_dtype: DType,
                 collect = True
                 ):
        self._num_of_agents = len(agent_list)
        # if time_step_spec.observation.shape.rank != 1:
        #     raise Exception('Cannot create single parallel model. Observation shape must have rank 1')
        super(MultiAgentSingleModelPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec,
        )
        self._policy = policy
        self._agent_list = agent_list
        self._collect = collect
        self.action = tf.function(self.action, jit_compile=config.USE_JIT)
        # self.gpu_split = len(tf.config.list_logical_devices('GPU')) + 1 == self._num_of_agents
        if collect:
            self._last_observation = tf.Variable(tf.zeros(shape=[1] + \
                                                                [self._num_of_agents] + \
                                                                self._agent_list[0].time_step_spec.observation.shape,
                                                          dtype=self.collect_data_spec.observation.dtype))
            self._last_action = tf.Variable(tf.zeros(shape=[1] + self.collect_data_spec.action.shape,
                                                     dtype=last_action_dtype))

    def get_last_trajectory(self, trajectory: Trajectory):
        return trajectory.replace(observation=self._last_observation.value(),
                                  action=self._last_action.value())

    def _action(self, time_step: ts.TimeStep,
                      policy_state: types.NestedTensor,
                      seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        #print('Tracing _action')
        obs = time_step.observation[0]
        agent_obs = [[]] * self._num_of_agents
        obs_info = [[]] * self._num_of_agents
        for i in range(self._num_of_agents):
            agent_obs[i], obs_info[i] = self._agent_list[i].get_observation(obs, time_step.step_type, self._collect)
        stacked_agent_obs = tf.stack(agent_obs)
        batched_observation = tf.expand_dims(stacked_agent_obs, axis=0)

        single_model_action_vector = self._policy.action(time_step._replace(observation=batched_observation))
        agent_action = [[]] * self._num_of_agents
        for i in range(self._num_of_agents):
            agent_action[i] = self._agent_list[i].get_action(single_model_action_vector.action[0][i],
                                                             stacked_agent_obs[i],
                                                             obs_info[i],
                                                             self._collect)
        stacked_agent_action = tf.stack(agent_action)
        if self._collect:
            # tf.print(batched_observation)
            self._last_observation.assign(batched_observation)
        if self._collect:
            self._last_action.assign(single_model_action_vector.action)
        batched_action = tf.expand_dims(stacked_agent_action, axis=0)
        return policy_step.PolicyStep(action=batched_action, state=(), info=())

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
        self.gpu_split = len(tf.config.list_logical_devices('GPU')) + 1 == self._num_of_agents
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
        # policy_action_list = [lambda: policy.action(time_step) for policy in self._policy_list]
        # action_tensor = tf.map_fn(lambda id: tf.switch_case(id,
        #                                                     policy_action_list).action,
        #                           self._agent_id_tensor,
        #                           fn_output_signature=tf.float32,
        #                           parallel_iterations=self._num_of_agents)
        action_list = [0.0 for _ in range(self._num_of_agents)]
        for i in range(self._num_of_agents):
            with tf.device(f'GPU{i}' if self.gpu_split else 'CPU:0'):
                action_list[i] = self._policy_list[i].action(time_step).action
        with tf.device(f'GPU{self._num_of_agents}' if self.gpu_split else 'CPU:0'):
            action_tensor = tf.stack(action_list, axis=0)
            # action_tensor = tf.parallel_stack(action_list)

            info = tf.fill([tf.shape(time_step.observation)[0], 1], self._global_step % MAX_BUFFER_SIZE)
            # info = tf.fill([tf.shape(time_step.observation)[0]], self._global_step % MAX_BUFFER_SIZE)
            if self._collect:
                self._global_step.assign_add(1)
            return policy_step.PolicyStep(action=tf.expand_dims(action_tensor, axis=0), state=(), info=info)

def create_single_model(network_list: List[Sequential], add_activation_layer = False):
    with tf.name_scope('SingleFunctionalModel'):
        num_of_networks = len(network_list)
        if hasattr(network_list[0].get_layer(index=0), 'input'):
            single_network_input_shape = network_list[0].get_layer(index=0).input.shape[1:]
            try:
                all([network.get_layer(index=0).input.shape[1:] == single_network_input_shape for network in network_list])
            except Exception:
                raise Exception('All inputs of agent networks must have the same shape')
        elif hasattr(network_list[0], 'input_tensor_spec'):
            single_network_input_shape = network_list[0].input_tensor_spec.shape
            try:
                all([network.input_tensor_spec.shape == single_network_input_shape for network in network_list])
            except Exception:
                raise Exception('All inputs of agent networks must have the same shape')
        # input_shape = [num_of_networks] + network_list[0].get_layer(index=0).input.shape[1:]
        input_shape = [num_of_networks] + single_network_input_shape
        # try:
        #     all([network.get_layer(index=0).input.shape[1:] == input_shape for network in network_list])
        # except Exception:
        #     raise Exception('All inputs of agent networks must have the same shape')
        input = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        output_list = []
        unstacked_inputs = tf.unstack(input, num=num_of_networks, axis=1)
        # unstacked_inputs = tf.keras.layers.Lambda(partial(tf.unstack, num=num_of_networks, axis=1), dtype=tf.float32)(input)
        for agent_id, network in enumerate(network_list):
            agent_network = network(unstacked_inputs[agent_id])[0]
            final_agent_network = agent_network if not add_activation_layer \
                                                else Activation('linear', dtype=tf.float32)(agent_network)
            output_list.append(final_agent_network)
        # output = tf.keras.layers.Concatenate(axis=1, dtype=tf.float32)(output_list)
        output = tf.stack(output_list, axis=1)
        # output = tf.keras.layers.Lambda(partial(tf.stack, axis=1))(output_list)
        model = MyNetwork(tf.keras.Model(inputs=input, outputs=output))
        return model

class MultipleAgents(tf.Module):
    # num_iterations = 24 * 100 * 100
    num_iterations = config.TRAIN_ITERATIONS
    num_eval_episodes = 20  # @param {type:"integer"}
    # num_eval_episodes = 1  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset
    # episode_eval_interval = 10
    # eval_interval = 1  # @param {type:"integer"} # 100 days in dataset

    initial_collect_episodes = 20 # @param {type:"integer"}
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

        ts = agents_list[0].time_step_spec
        # tensor_spec.add_outer_dim(agents_list[0].time_step_spec, dim=3)
        multi_agent_time_step_spec = TimeStep(step_type=ts.step_type,
                                              # step_type=tensor_spec.add_outer_dim(ts.step_type, 1),
                                              discount=ts.discount,
                                              # discount=tensor_spec.add_outer_dim(ts.discount, 1),
                                              # reward=ts.reward,
                                              reward=tensor_spec.add_outer_dim(ts.reward,
                                                                               dim=self._number_of_agents),
                                              observation=tensor_spec.add_outer_dim(ts.observation,
                                                                                    dim=self._number_of_agents))

        # multi_agent_action_spec = tensor_spec.add_outer_dims_nest(agents_list[0].action_spec, (self._number_of_agents,))
        multi_agent_action_spec = tensor_spec.add_outer_dims_nest(agents_list[0].action_spec, (self._number_of_agents,))

        self._collect_data_context = data_converter.DataContext(
            time_step_spec=multi_agent_time_step_spec,
            action_spec=multi_agent_action_spec,
            info_spec=(),
        )

        collect_ts = self._collect_data_context.time_step_spec
        self._training_data_spec = data_converter.DataContext(
            time_step_spec=collect_ts._replace(step_type=tensor_spec.add_outer_dim(collect_ts.step_type, 1),
                                               discount=tensor_spec.add_outer_dim(collect_ts.discount, 1)),
            action_spec=self._collect_data_context.action_spec,
            info_spec=self._collect_data_context.info_spec,).trajectory_spec

        self._total_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        # self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='GLOBAL_STEP_MA')
        self.gpu_split = len(tf.config.list_logical_devices('GPU')) + 1 == self._number_of_agents
        device = f'/GPU:{self._number_of_agents}' if len(tf.config.list_physical_devices('GPU')) else "/device:CPU:0"
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(self.collect_data_spec,
                                                                            device=device,
                                                                            batch_size=self.train_env.batch_size,
                                                                            max_length=self.replay_buffer_capacity)
        self.checkpoint = common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=1,
            replay_buffer=self.replay_buffer,
            # global_step=self.global_step,
        )
        if self.checkpoint.checkpoint_exists:
            print("Checkpoint found on specified folder. Continuing from there...")

        single_model_q_network = create_single_model([agent._q_network for agent in agents_list])
        single_model_target_q_network = create_single_model([agent._target_q_network for agent in agents_list])
        # multi_agent_time_step_spec = tensor_spec.add_outer_dim(agents_list[0].time_step_spec, self._number_of_agents)
        # multi_agent_action_spec = tensor_spec.add_outer_dim(agents_list[0].action_spec, self._number_of_agents)

        #Monkey Patching the function which computes the discounted rewards to work with our multiple agents environment
        #(It will treat the reward tensor as having 3 dimensions (B, T) plus another for the number of agents
        tf_agents.utils.value_ops.discounted_return = my_discounted_return

        #Monkey Patching the function which gathers the q_values specified by the policy actions to be able to work with
        #the extra dimension of the q values vector (just adds a dimension for the individual agents actions)
        tf_agents.utils.common.index_with_actions = my_index_with_actions

        #Monkey Patching the function tha creates transitions out of trajectories because it doesn't support 1 dimensional
        #rewards (It needed some changes). Also, it's used by dqn agents during initialization so it has to be patched
        #after the single agents creation
        tf_agents.trajectories.trajectory.to_n_step_transition = my_to_n_step_transition


        kwargs = {
            'num_of_agents': self._number_of_agents, #needed to scale the gradients, because otherwise the number of outputs is interpreted as batchsize
            'time_step_spec': multi_agent_time_step_spec,
            'action_spec': multi_agent_action_spec,
            'training_data_spec': self._training_data_spec,
            # 'action_spec': tensor_spec.add_outer_dim(agents_list[0].action_spec, 1),
            # 'action_spec': agents_list[0].action_spec,
            'q_network': single_model_q_network,
            'target_q_network': single_model_target_q_network,
            'optimizer': tf.keras.optimizers.Adam(learning_rate=3e-4),
            # 'td_errors_loss_fn': common.element_wise_squared_loss,
            # 'epsilon_greedy': 0.2,
            'epsilon_greedy': None,
            'boltzmann_temperature': 0.8,
            'target_update_tau': 0.2,
            'target_update_period': 1000,
        }
        self._multi_dqn_agent = MultiDdqnAgent(**kwargs)

        # x = tensor_spec.sample_spec_nest(self.collect_data_spec, outer_dims=(1, 2))
        # x = x.replace(step_type=tf.ones_like(x.step_type, dtype=tf.int64))
        # self._multi_dqn_agent.train(x)


        self.collect_policy = MultiAgentSingleModelPolicy(self._multi_dqn_agent.collect_policy,
                                                          self._agent_list,
                                                          self.train_env.time_step_spec(),
                                                          self.train_env.action_spec(),
                                                          (),
                                                          (),
                                                          tf.int64,
                                                          True,)
        self.policy = MultiAgentSingleModelPolicy(self._multi_dqn_agent.policy,
                                                  self._agent_list,
                                                  self.eval_env.time_step_spec(),
                                                  self.eval_env.action_spec(),
                                                  (),
                                                  (),
                                                  tf.int64,
                                                  False)

        self._initial_policy = self.wrap_policy(initial_collect_policy, True) if initial_collect_policy else None

        steps = self.num_eval_episodes * 24
        self._metric = tf_metrics.AverageReturnMetric(batch_size=self.eval_env.batch_size,
                                                      buffer_size=steps)
        self._eval_driver = TFDriver(self.eval_env,
                                     self.policy,
                                     [self._metric],
                                     max_episodes=self.num_eval_episodes,
                                     disable_tf_function=False)
        # self._eval_driver.run = tf.function(self._eval_driver.run, jit_compile=True)



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
        if not config.USE_JIT:
            best_avg_return = self.eval_policy()
            print(best_avg_return)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(10)
        iterator = iter(dataset)

        if self.replay_buffer.num_frames() < 24 * self.initial_collect_episodes:
            self.train_env.hard_reset()
            for agent in self._agent_list:
                agent.reset_collect_steps()
            TFDriver(self.train_env,
                     self.collect_policy,
                     [lambda traj: self.replay_buffer.add_batch(self.collect_policy.get_last_trajectory(traj))],
                     # [lambda traj: self.replay_buffer.add_batch(traj)],
                     max_episodes=self.initial_collect_episodes,
                     disable_tf_function=False).run(self.train_env.reset())

        # @tf.function(jit_compile=True)
        # def _train_on_experience(experience):
        #     train_list = [lambda: agent.train(experience).loss for agent in self._agent_list]
        #     train_loss = tf.map_fn(lambda id: tf.switch_case(id,
        #                                                      train_list),
        #                            tf.range(self._number_of_agents, dtype=tf.int32),
        #                            fn_output_signature=tf.float32,
        #                            parallel_iterations=self._number_of_agents)
        #     self._total_loss.assign_add(train_loss / tf.cast(24 * self.epochs, tf.float32))
        # @tf.function(jit_compile=True)
        # def _train_on_experience(experience):
        #     train_loss_list = [0.0 for _ in range(self._number_of_agents)]
        #     for i in range(self._number_of_agents):
        #         with tf.device(f'GPU{i}' if self.gpu_split else 'CPU:0'):
        #             train_loss_list[i] = self._agent_list[i].train(experience).loss
        #     with tf.device(f'GPU{self.number_of_agents}' if self.gpu_split else 'CPU:0'):
        #         train_loss = tf.stack(train_loss_list, axis=0)
        #         self._total_loss.assign_add(train_loss / tf.cast(24 * self.epochs, tf.float32))

        def train_step():
            # experience, _ = next(iterator)
            experience, _ = self.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=2)
            # _train_on_experience(experience)
            loss = self._multi_dqn_agent.train(experience).loss
            self._total_loss.assign_add(loss / tf.cast(24 * self.epochs, tf.float32))

        collect_driver = TFDriver(self.train_env,
                                  self.collect_policy,
                                  [lambda traj: self.replay_buffer.add_batch(self.collect_policy.get_last_trajectory(traj)),
                                   lambda traj: train_step()],
                                  max_episodes=self.epochs,
                                  disable_tf_function=False)
        # @tf.function
        def train_epoch():
            self.train_env.hard_reset()
            for agent in self._agent_list:
                agent.reset_collect_steps()
            collect_driver.run(self.train_env.reset())
            if config.USE_JIT:
                return tf.constant([0.0])
            else:
                return self.eval_policy()

        # collect_driver.run = tf.function(collect_driver.run, jit_compile=True)
        time_step = self.train_env.reset()
        # epoch_train_st = 0.0
        loss_acc = [0.0] * self._number_of_agents
        epoch_st = time.time()


        for i in range(1, self.num_iterations+1):
            avg_return = train_epoch()
            print('Epoch: ', i, '            Avg_return = ', avg_return.numpy())
            print('Avg Train Loss: ', self._total_loss.value().numpy())
            epoch_et = time.time()
            self._total_loss.assign(self._total_loss * tf.constant(0.0, tf.float32))
            print('Epoch duration: ', epoch_et - epoch_st)
            epoch_st = epoch_et
            self.returns.append(avg_return)
            if config.KEEP_BEST:
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    # self.global_step.assign(self.collect_policy.global_step)
                    # self.checkpoint.save(self.global_step)
                    train_counter = self._multi_dqn_agent.train_step_counter
                    self.checkpoint.save(train_counter)
                    for agent in self._agent_list:
                        agent.checkpoint_save(train_counter)
            elif i % 100 == 0:
                # self.global_step.assign(self.collect_policy.global_step)
                train_counter = self._multi_dqn_agent.train_step_counter
                self.checkpoint.save(train_counter)
                for agent in self._agent_list:
                    agent.checkpoint_save(train_counter)
        self.checkpoint.save(self._multi_dqn_agent.train_step_counter)



    def wrap_policy(self, policy_list: list, collect: bool):
        if len(policy_list) != self._number_of_agents:
           raise Exception('Policies List should be a list of size equal to the number of agents, and have different,'
                           'policy objects')
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
            time_step = self.eval_env.reset()
            driver.run(time_step)
            return self._metric.result()





