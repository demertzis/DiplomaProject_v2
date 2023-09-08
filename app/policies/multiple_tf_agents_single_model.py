import shutil
import time
from typing import List, Optional

import tensorflow as tf
import tf_agents
from keras.layers import Activation
from tensorflow import DType
from tf_agents.agents import TFAgent, data_converter
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import Sequential
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts, TimeStep, PolicyStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
from tf_agents.utils import common

import config
from app.abstract.multi_dqn import MultiDdqnAgent
from app.abstract.tf_unified_agents_single_model import UnifiedAgent
from app.abstract.utils import MyNetwork, my_discounted_return, my_index_with_actions, my_to_n_step_transition, \
    MyCheckpointer
from app.models.tf_pwr_env_2 import TFPowerMarketEnv
from config import MAX_BUFFER_SIZE


class MultiAgentSingleModelPolicy(TFPolicy):
    def __init__(self,
                 policy,
                 agent_list,
                 time_step_spec,
                 action_spec,
                 policy_state_spec,
                 info_spec,
                 last_action_dtype: DType,
                 collect=True
                 ):
        if isinstance(agent_list, UnifiedAgent):
            self._use_unified_agent = True
            self._num_of_agents = agent_list._num_of_agents
            self._agent_list = [agent_list]
        else:
            self._use_unified_agent = False
            self._num_of_agents = len(agent_list)
            self._agent_list = agent_list
        # if time_step_spec.observation.shape.rank != 1:
        #     raise Exception('Cannot create single parallel model. Observation shape must have rank 1')
        super(MultiAgentSingleModelPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec,
        )
        self._policy = policy
        self._collect = collect
        self.action = tf.function(self.action, jit_compile=config.USE_JIT)
        # self.gpu_split = len(tf.config.list_logical_devices('GPU')) + 1 == self._num_of_agents
        if collect:
            if isinstance(agent_list, UnifiedAgent):
                single_agent_observation_shape = self._agent_list[0].agent_list[0].time_step_spec.observation.shape
            else:
                single_agent_observation_shape = self._agent_list[0].time_step_spec.observation.shape
            self._last_observation = tf.Variable(tf.zeros(shape=[1, self._num_of_agents] +\
                                                                single_agent_observation_shape,
                                                          dtype=self.collect_data_spec.observation.dtype))
            # self._last_action = tf.Variable(tf.zeros(shape=[1] + self.collect_data_spec.action.shape,
            #                                          dtype=last_action_dtype))
            self._last_action = tf.Variable(tf.zeros(shape=[1, self._num_of_agents],
                                                     dtype=last_action_dtype))

    def get_last_trajectory(self, trajectory: Trajectory):
        return trajectory.replace(observation=self._last_observation.value(),
                                  action=self._last_action.value())

    def _action(self, time_step: ts.TimeStep,
                policy_state: types.NestedTensor,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        # print('Tracing _action')
        obs = time_step.observation[0]
        agent_obs = [[]] * len(self._agent_list)
        obs_info = [[]] * len(self._agent_list)
        for i in range(len(self._agent_list)):
            agent_obs[i], obs_info[i] = self._agent_list[i].get_observation(obs, time_step.step_type, self._collect)
        stacked_agent_obs = tf.stack(agent_obs, axis=0)
        if not self._use_unified_agent:
            batched_observation = tf.expand_dims(stacked_agent_obs, axis=0)
        else:
            batched_observation = stacked_agent_obs
        single_model_action_vector = self._policy.action(time_step._replace(observation=batched_observation))
        agent_action = [[]] * len(self._agent_list)
        for i in range(len(self._agent_list)):
            agent_action[i] = self._agent_list[i].get_action(single_model_action_vector.action[0][i],
                                                             stacked_agent_obs[i],
                                                             obs_info[i],
                                                             self._collect)
        stacked_agent_action = tf.stack(agent_action, axis=0)
        if not self._use_unified_agent:
            batched_action = tf.expand_dims(stacked_agent_action, axis=0)
        else:
            batched_action = stacked_agent_action
        if self._collect:
            self._last_observation.assign(batched_observation)
        if self._collect:
            self._last_action.assign(single_model_action_vector.action)

        return policy_step.PolicyStep(action=batched_action, state=(), info=())


def create_single_model(network_list: List[Sequential], add_activation_layer=False):
    with tf.name_scope('SingleFunctionalModel'):
        num_of_networks = len(network_list)
        if hasattr(network_list[0].get_layer(index=0), 'input'):
            single_network_input_shape = network_list[0].get_layer(index=0).input.shape[1:]
            try:
                all([network.get_layer(index=0).input.shape[1:] == single_network_input_shape for network in
                     network_list])
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

    initial_collect_episodes = 20  # @param {type:"integer"}
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
                 ckpt_dir="new_checkpoints",
                 initial_collect_policy: Optional = None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.ckpt_dir = ckpt_dir
        self.returns = []
        self._unified_agent = UnifiedAgent(agents_list)
        self._agent_list: List[TFAgent] = self._unified_agent.agent_list
        self._number_of_agents = len(self._agent_list)

        ts = self._agent_list[0].time_step_spec
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
        multi_agent_action_spec = tensor_spec.add_outer_dims_nest(self._agent_list[0].action_spec, (self._number_of_agents,))

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
            info_spec=self._collect_data_context.info_spec, ).trajectory_spec

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

        single_model_q_network = create_single_model([agent._q_network for agent in self._agent_list])
        single_model_target_q_network = create_single_model([agent._target_q_network for agent in self._agent_list])
        # multi_agent_time_step_spec = tensor_spec.add_outer_dim(agents_list[0].time_step_spec, self._number_of_agents)
        # multi_agent_action_spec = tensor_spec.add_outer_dim(agents_list[0].action_spec, self._number_of_agents)

        # Monkey Patching the function which computes the discounted rewards to work with our multiple agents environment
        # (It will treat the reward tensor as having 3 dimensions (B, T) plus another for the number of agents
        tf_agents.utils.value_ops.discounted_return = my_discounted_return

        # Monkey Patching the function which gathers the q_values specified by the policy actions to be able to work with
        # the extra dimension of the q values vector (just adds a dimension for the individual agents actions)
        tf_agents.utils.common.index_with_actions = my_index_with_actions

        # Monkey Patching the function tha creates transitions out of trajectories because it doesn't support 1 dimensional
        # rewards (It needed some changes). Also, it's used by dqn agents during initialization so it has to be patched
        # after the single agents creation
        tf_agents.trajectories.trajectory.to_n_step_transition = my_to_n_step_transition

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=24000,
            staircase=True,
            decay_rate=0.9)

        kwargs = {
            'num_of_agents': self._number_of_agents,
            # needed to scale the gradients, because otherwise the number of outputs is interpreted as batchsize
            'time_step_spec': multi_agent_time_step_spec,
            'action_spec': multi_agent_action_spec,
            'training_data_spec': self._training_data_spec,
            # 'action_spec': tensor_spec.add_outer_dim(agents_list[0].action_spec, 1),
            # 'action_spec': agents_list[0].action_spec,
            'q_network': single_model_q_network,
            'target_q_network': single_model_target_q_network,
            # 'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-4),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate),
            # 'td_errors_loss_fn': common.element_wise_squared_loss,
            # 'epsilon_greedy': 0.1,
            'epsilon_greedy': None,
            'boltzmann_temperature': 0.4,
            'target_update_tau': 0.1,
            'target_update_period': 2400,
        }
        self._multi_dqn_agent = MultiDdqnAgent(**kwargs)

        # x = tensor_spec.sample_spec_nest(self.collect_data_spec, outer_dims=(1, 2))
        # x = x.replace(step_type=tf.ones_like(x.step_type, dtype=tf.int64))
        # self._multi_dqn_agent.train(x)

        self.collect_policy = MultiAgentSingleModelPolicy(self._multi_dqn_agent.collect_policy,
                                                          # self._agent_list,
                                                          self._unified_agent,
                                                          self.train_env.time_step_spec(),
                                                          self.train_env.action_spec(),
                                                          (),
                                                          (),
                                                          tf.int64,
                                                          True, )
        self.policy = MultiAgentSingleModelPolicy(self._multi_dqn_agent.policy,
                                                  # self._agent_list,
                                                  self._unified_agent,
                                                  self.eval_env.time_step_spec(),
                                                  self.eval_env.action_spec(),
                                                  (),
                                                  (),
                                                  tf.int64,
                                                  False)
        # saver = tf_agents.policies.PolicySaver(self.policy, batch_size=self.eval_env.batch_size)
        # Network.__call__ = __call__
        # saver.save("saved_policies")
        self.best_checkpoint = MyCheckpointer(
            ckpt_dir='best_' + ckpt_dir,
            max_to_keep=1,
            policy=self.policy
        )

        self._temp_ckpt_dir = 'temp_' + ckpt_dir
        self.temp_checkpoint = MyCheckpointer(
            ckpt_dir=self._temp_ckpt_dir,
            max_to_keep=1,
            policy=self.policy
        )
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

    def train(self):
        print('Latest trained policy evaluation: {}'.format(self.eval_policy().numpy()))
        best_avg_return = self.eval_policy(best=True)
        print('Best policy evaluation: {}'.format(best_avg_return.numpy()))
        # input('Press Enter to continue...')

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

        def train_step():
            # experience, _ = next(iterator)
            experience, _ = self.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=2)
            # _train_on_experience(experience)
            loss = self._multi_dqn_agent.train(experience).loss
            self._total_loss.assign_add(loss / tf.cast(24 * self.epochs, tf.float32))

        collect_driver = TFDriver(self.train_env,
                                  self.collect_policy,
                                  [lambda traj: self.replay_buffer.add_batch(
                                      self.collect_policy.get_last_trajectory(traj)),
                                   lambda traj: train_step()],
                                  max_episodes=self.epochs,
                                  disable_tf_function=False)

        @tf.function
        def train_epoch():
            self.train_env.hard_reset()
            for agent in self._agent_list:
                agent.reset_collect_steps()
            collect_driver.run(self.train_env.reset())
            if config.USE_JIT:
                return tf.constant([0.0])
            else:
                return self.eval_policy()

        # loss_acc = [0.0] * self._number_of_agents
        epoch_st = time.time()

        for i in range(1, self.num_iterations + 1):
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
                    # self.checkpoint.save(train_counter)
                    self.best_checkpoint.save(train_counter)
                    # for agent in self._agent_list:
                    #     agent.checkpoint_save(train_counter, True)
            elif i % 100 == 0:
                # self.global_step.assign(self.collect_policy.global_step)
                train_counter = self._multi_dqn_agent.train_step_counter
                self.checkpoint.save(train_counter)
                for agent in self._agent_list:
                    agent.checkpoint_save(train_counter)
        train_counter = self._multi_dqn_agent.train_step_counter
        self.checkpoint.save(train_counter)
        for agent in self._agent_list:
            agent.checkpoint_save(train_counter)

    def wrap_policy(self, policy_list: List[TFPolicy], collect: bool):
        if len(policy_list) != self._number_of_agents:
            raise Exception('Policies List should be a list of size equal to the number of agents, and have different,'
                            'policy objects')
        # @tf.function(jit_compile=config.USE_JIT)
        num_of_agents = self._number_of_agents

        class TempPolicy():
            def action(self,
                       time_step: ts.TimeStep,
                       policy_state: types.NestedTensor = (),
                       seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
                actions = [[]] * num_of_agents
                for i in range(num_of_agents):
                    agent_time_step = time_step._replace(observation=time_step.observation[:, i])
                    actions[i] = policy_list[i].action(agent_time_step, policy_state, seed).action
                return PolicyStep(action=tf.stack(actions, axis=1))

        policy = TempPolicy()
        env = self.train_env if collect else self.eval_env
        return MultiAgentSingleModelPolicy(policy,
                                           self._agent_list,
                                           env.time_step_spec(),
                                           env.action_spec(),
                                           (),
                                           (),
                                           tf.int64,
                                           collect)

    # @tf.function
    def eval_policy(self, policy_list: Optional[List] = None, best=False):
        # print('Tracing eval_policy')
        if policy_list and best:
            raise Exception('Can only evaluate either a policy list or a saved policy not both')
        if best:
            self.temp_checkpoint.save(global_step=tf.constant(0, tf.int64))
            self.best_checkpoint.initialize_or_restore()
            if self.best_checkpoint.checkpoint_exists:
                print('Found Best Policy. Continuing from there...')
        self._metric.reset()
        for agent in self._agent_list:
            agent.reset_eval_steps()
        self.eval_env.hard_reset()
        if policy_list is None:
            self._eval_driver.run(self.eval_env.reset())
            if best:
                self.temp_checkpoint.initialize_or_restore()
                try:
                    shutil.rmtree(self._temp_ckpt_dir)
                    print(f"Deleted directory: {self._temp_ckpt_dir}")
                except Exception as e:
                    print(f"Error deleting directory: {e}")
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
