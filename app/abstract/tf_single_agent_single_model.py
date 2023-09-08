import math
from typing import Callable, List

import tensorflow as tf
from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.typing.types import SpecTensorOrArray
from tf_agents.utils.common import Checkpointer

from app.abstract.utils import MyCheckpointer
from app.models.tf_parking_10 import Parking
from app.utils import generate_vehicles_constant_shape
from config import VEHICLE_BATTERY_CAPACITY, NUM_OF_ACTIONS


def create_single_agent(cls: type,
                        vehicle_distribution: List,
                        name: str,
                        # num_of_agents: int,
                        ckpt_dir: str,
                        buffer_max_size: int,
                        num_of_actions: int = NUM_OF_ACTIONS,
                        capacity_train_garage: int = 100,
                        capacity_eval_garage: int = 100,
                        coefficient_function: Callable = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5,
                        *args,
                        **kwargs, ):
    class_list = [cls]
    temp = cls
    while temp.__bases__:
        class_list.append(temp.__bases__[0])
        temp = temp.__bases__[0]

    if not any([i == TFAgent for i in class_list]):
        raise Exception('Provided Class is not TFAgent.'
                        'Class provided: {}'.format(cls.__class__.__name__))

    class SingleAgent(cls):
        def __init__(self,
                     vehicle_distribution: List,
                     coefficient_function: Callable,
                     ckpt_dir: str,
                     buffer_max_size: int,
                     num_of_actions: int,
                     capacity_train_garage: int = 100,
                     capacity_eval_garage: int = 200,
                     name: str = "DefaultAgent",
                     *args,
                     **kwargs):
            self._agent_id = int("".join(item for item in list(filter(str.isdigit, name))))
            if not self._agent_id:
                raise Exception('Agent _name must have an integer'
                                ' to denote the agents unique id.'
                                'For example: Agent_1')
            # Parking fields. Can be removed in a different setting
            self._num_of_actions = num_of_actions
            self._capacity_train_garage = capacity_train_garage
            self._capacity_eval_garage = capacity_eval_garage
            self._train_parking = Parking(capacity_train_garage, 'train')
            self._eval_parking = Parking(capacity_eval_garage, 'eval')
            # self._generate_vehicles_train = generate_vehicles(coefficient_function)
            self._coefficient_function = coefficient_function
            self._generate_vehicles_train = generate_vehicles_constant_shape(coefficient_function,
                                                                             self._capacity_train_garage)
            # with open('data/vehicles_old.json') as file:
            #     vehicles = list(json.load(file))
            # self._eval_vehicles = tf.ragged.constant(vehicle_distribution)
            self._eval_vehicles = tf.constant(vehicle_distribution)
            # self._private_observations = tf.Variable(tf.zeros([buffer_max_size, 21], dtype=tf.float16),
            #                                          shape=[buffer_max_size, 21],
            #                                          dtype=tf.float16,
            #                                          trainable=False)
            # self._private_observations = tf.Variable(tf.zeros([buffer_max_size, 21], dtype=tf.float32),
            #                                          shape=[buffer_max_size, 21],
            #                                          dtype=tf.float32,
            #                                          trainable=False,
            #                                          name=name + ': private observations')
            # self._private_actions = tf.Variable(tf.zeros([buffer_max_size], tf.int64),
            #                                     shape=[buffer_max_size],
            #                                     dtype=tf.int64,
            #                                     trainable=False,
            #                                     name=name + ': private actions')
            # self._private_index = tf.Variable(0,
            #                                   dtype=tf.int64,
            #                                   trainable=False,
            #                                   name=name + ': private index')
            # self._buffer_max_size = buffer_max_size
            # TODO decide on weather it's needed
            # TODO remove avg vehicle list, there should not be access to global variables

            self._collect_steps = tf.Variable(-1,
                                              dtype=tf.int64,
                                              trainable=False,
                                              name=name + ': time_of_day')
            self._eval_steps = tf.Variable(-1,
                                           dtype=tf.int64,
                                           trainable=False,
                                           name=name + ': eval steps')
            # self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name=name + ': global step')
            # super(cls, self).__init__(*args, train_step_counter=self.global_step, **kwargs)
            super(cls, self).__init__(*args, **kwargs)
            self._name = name
            self._num_of_multi_agent_observation_elements = self.time_step_spec.observation.shape[0] - 21
            self.checkpointer = Checkpointer(
                ckpt_dir='/'.join([ckpt_dir, self._name]),
                max_to_keep=1,
                agent=self,
                # policy=self.policy,
            )
            self.best_checkpointer = MyCheckpointer(
                ckpt_dir='/'.join(['best_' + ckpt_dir, self._name]),
                max_to_keep=1,
                policy=self.policy,
                # policy=self.policy,
            )
            # self.policy.action = self._action_wrapper(self.policy.action, False)
            # self.collect_policy.action = self._action_wrapper(self.collect_policy.action, True)

            # buffer_time_step_spec = TimeStep(
            #         step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.int64, minimum=0, maximum=2),
            #         discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
            #         reward=tensor_spec.TensorSpec(shape=(num_of_agents,), dtype=np.float32),
            #         observation=tensor_spec.BoundedTensorSpec(shape=(13,), dtype=np.float32, minimum=-1., maximum=1.),
            #     )
            # buffer_info_spec = tensor_spec.TensorSpec(shape=(1,),
            #                                           dtype=tf.int64)
            # self._collect_data_context = data_converter.DataContext(
            #     time_step_spec=buffer_time_step_spec,
            #     action_spec=(),
            #     info_spec=buffer_info_spec,
            # )

        def checkpoint_save(self, global_step, best=False):
            if best:
                self.best_checkpointer.save(global_step)
            else:
                self.checkpointer.save(global_step)

        def reset_collect_steps(self):
            self._collect_steps.assign(-1)

        def reset_eval_steps(self):
            self._eval_steps.assign(-1)

        @property
        def private_index(self):
            return self._private_index.value().numpy()

        @property
        def coefficient_function(self):
            return self._coefficient_function

        @private_index.setter
        def private_index(self, index):
            self._private_index.assign(index)

        @property
        def train_vehicles_generator(self):
            return self._generate_vehicles_train

        # @tf.function
        # def _add_new_cars(self, train_mode=True): # TODO decide on way to choose between distributions
        #     #print('Tracing add_new_cars')
        #     if train_mode:
        #         vehicles = self._generate_vehicles_train(self._time_of_day)
        #         # capacity = self._train_parking.get_capacity()
        #     else:
        #         vehicles = self._eval_vehicles[self._eval_steps].to_tensor()
        #         vehicles = tf.reshape(vehicles, [-1, 3])
        #         # capacity = self._eval_parking.get_capacity()
        #     max_min_charges = tf.repeat(tf.constant([[VEHICLE_BATTERY_CAPACITY, 0.]], tf.float32),
        #                                 repeats=tf.shape(vehicles)[0],
        #                                 axis=0)
        #     vehicles = tf.concat((vehicles, max_min_charges), axis=1)
        #     vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
        #     if train_mode:
        #         self._train_parking.assign_vehicles(vehicles)
        #     else:
        #         self._eval_parking.assign_vehicles(vehicles)

        # @tf.function(jit_compile=True)
        def _add_new_cars(self, train_mode=True):  # TODO decide on way to choose between distributions
            # print('Tracing add_new_cars')
            if train_mode:
                # vehicles = self._generate_vehicles_train(self._collect_steps % tf.constant(24, tf.int64))
                max_min_charges = tf.repeat([[VEHICLE_BATTERY_CAPACITY, 0.0] + [0.0] * 8],
                                            self._capacity_train_garage,
                                            axis=0)
                vehicles = tf.concat((self._generate_vehicles_train(self._collect_steps % tf.constant(24, tf.int64)),
                                      max_min_charges),
                                     axis=1)
            else:
                # vehicles = self._eval_vehicles[self._eval_steps]
                max_min_charges = tf.repeat([[VEHICLE_BATTERY_CAPACITY, 0.0] + [0.0] * 8],
                                            self._capacity_eval_garage,
                                            axis=0)
                vehicles = tf.concat((self._eval_vehicles[self._eval_steps],
                                      max_min_charges),
                                     axis=1)
            # max_min_charges = tf.repeat(tf.constant([[VEHICLE_BATTERY_CAPACITY, 0.]], tf.float32),
            #                             repeats=capacity,
            #                             axis=0)
            # max_min_charges = tf.constant([[VEHICLE_BATTERY_CAPACITY, 0.0]] * capacity, tf.float32)
            # max_min_charges = tf.where(tf.less(0.0, vehicles[..., 2:3]), [VEHICLE_BATTERY_CAPACITY, 0.0], [0.0, 0.0])
            # vehicles = tf.concat((vehicles, max_min_charges), axis=1)
            # vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
            if train_mode:
                self._train_parking.assign_vehicles(vehicles)
            else:
                self._eval_parking.assign_vehicles(vehicles)

        # @tf.function
        def _train(self, experience: types.NestedTensor,
                   weights: types.Tensor) -> LossInfo:
            print('Tracing train')
            return super()._train(self.preprocess_sequence(experience), weights)

        def get_action(self, step: SpecTensorOrArray, augmented_obs: SpecTensorOrArray, parking_fields, collect=True):
            # if collect:
            #     indices_tensor = (tf.constant([[0]], tf.int64) + self._private_index) % self._buffer_max_size
            #     self._private_index.assign_add(tf.constant(1, tf.int64))
            #     self._private_observations.scatter_nd_update(indices_tensor, tf.expand_dims(augmented_obs[13:], axis=0))
            #     self._private_actions.scatter_nd_update(indices_tensor, tf.expand_dims(step, axis=0))
            load = self._get_load(step,
                                  augmented_obs,
                                  parking_fields,
                                  collect)
            return load

        def get_observation(self, observation: SpecTensorOrArray, step_type: SpecTensorOrArray,
                            collect=True) -> SpecTensorOrArray:
            """
            Adds cars and returns the parking observation concatenated at the end of the environment observation.
            Should be used exactly once per step in the environment.
            """
            if collect:
                self._collect_steps.assign_add(tf.clip_by_value(tf.negative(step_type[0]) + \
                                                                tf.constant(2, tf.int64),
                                                                tf.constant(0, tf.int64),
                                                                tf.constant(1, tf.int64)))
            else:
                self._eval_steps.assign_add(tf.clip_by_value(tf.negative(step_type[0]) + \
                                                             tf.constant(2, tf.int64),
                                                             tf.constant(0, tf.int64),
                                                             tf.constant(1, tf.int64)))
            self._add_new_cars(collect)
            parking_obs, parking_fields = self._get_parking_observation(collect)
            return tf.concat((observation, parking_obs), axis=0), parking_fields

        # def _action_wrapper(self, action, collect=True):
        #     """
        #     Wraps the action method of a policy to allow it to consume
        #     timesteps from the single buffer, augmenting the observation
        #     with the saved garage state (or not if it's called during training)
        #     """
        #     # @tf.function(jit_compile=True)
        #     def wrapped_action_eval(time_step: TimeStep,
        #                             policy_state: types.NestedTensor = (),
        #                             seed: Optional[types.Seed] = None,) -> PolicyStep:
        #         #print('Tracing wrapped_action_eval')
        #         try:
        #             # time_step = nest_utils.prune_extra_keys(self.policy.time_step_spec, time_step)
        #             # policy_state = nest_utils.prune_extra_keys(self.policy.policy_state_spec,
        #             #                                            policy_state)
        #             # nest_utils.assert_same_structure(
        #             #     time_step,
        #             #     self.policy.time_step_spec,
        #             #     message='time_step and time_step_spec structures do not match')
        #             # return action(time_step,
        #             return action(time_step._replace(observation=tf.cast(time_step.observation, tf.float32)),
        #                           policy_state,
        #                           seed)
        #         except (TypeError, ValueError):
        #             # if ~time_step.is_last():
        #             #     self._add_new_cars(False)
        #             #     self._eval_steps.assign_add(1)
        #             # self._eval_steps.assign_add(tf.where(tf.squeeze(time_step.is_last()),
        #             #                                      tf.constant(0, tf.int64),
        #             #                                      tf.constant(1, tf.int64)))
        #             self._eval_steps.assign_add(tf.clip_by_value(tf.negative(time_step.step_type[0]) +\
        #                                                           tf.constant(2, tf.int64),
        #                                                          tf.constant(0, tf.int64),
        #                                                          tf.constant(1, tf.int64)))
        #             self._add_new_cars(False)
        #             # index = tf.where(tf.squeeze(time_step.is_last()), 0, 1)
        #             # self._eval_steps.assign_add(tf.cast(index, tf.int64))
        #             # def add_cars():
        #             #     self._add_new_cars(False)
        #             # tf.switch_case(index, [tf.no_op, add_cars])
        #             # tf.print(time_step.observation)
        #             parking_obs = tf.expand_dims(self._get_parking_observation(False), axis=0)
        #             augmented_obs = tf.concat((time_step.observation,
        #                                        parking_obs),
        #                                       axis=1)
        #             reward = time_step.reward[..., self._agent_id - 1]
        #             # new_time_step = time_step._replace(observation=augmented_obs,
        #             new_time_step = time_step._replace(observation=tf.cast(augmented_obs, tf.float32),
        #                                                reward=reward)
        #             step = action(new_time_step,
        #                           policy_state,
        #                           seed,)
        #             load = self._get_load(tf.squeeze(step.action),
        #                                   tf.squeeze(augmented_obs),
        #                                   False)
        #             # load = tf.expand_dims(load, axis=0)
        #             # load = tf.reshape(load, [1, -1])
        #             # return step.replace(action=tf.reshape(load, shape=[1]))
        #             # load = tf.reshape(load, [1, -1])
        #             return step.replace(action=load)
        #
        #     @tf.function(jit_compile=True)
        #     def wrapped_action_collect(time_step: TimeStep,
        #                                policy_state: types.NestedTensor = (),
        #                                seed: Optional[types.Seed] = None,) -> PolicyStep:
        #         #print('Tracing wrapped_action_collect')
        #         # if ~time_step.is_last():
        #         #     self._add_new_cars(True)
        #         #     self._time_of_day.assign_add(1)
        #         # else:
        #         #     self._time_of_day.assign(0)
        #         # self._collect_steps.assign_add(tf.where(tf.squeeze(time_step.is_last()),
        #         #                                tf.constant(0, tf.int64),
        #         #                                tf.constant(1, tf.int64)))
        #         self._collect_steps.assign_add(tf.clip_by_value(tf.negative(time_step.step_type[0]) +\
        #                                                          tf.constant(2, tf.int64),
        #                                                         tf.constant(0, tf.int64),
        #                                                         tf.constant(1, tf.int64)))
        #         self._add_new_cars(True)
        #         parking_obs = tf.expand_dims(self._get_parking_observation(True), axis=0)
        #         augmented_obs = tf.concat((time_step.observation,
        #                                    parking_obs),
        #                                   axis=1)
        #         reward = time_step.reward[..., self._agent_id - 1]
        #         # new_time_step = time_step._replace(observation=augmented_obs,
        #         new_time_step = time_step._replace(observation=tf.cast(augmented_obs, tf.float32),
        #                                            reward=reward)
        #         step = action(new_time_step,
        #                       policy_state,
        #                       seed,)
        #         time_step_batch_size = tf.shape(time_step.observation, out_type=tf.int64)[0]
        #         # indices_tensor = tf.reshape((tf.range(time_step_batch_size) +
        #         #                              self._private_index) %
        #         #                              self._buffer_max_size,
        #         #                             [-1, 1])
        #         indices_tensor = tf.expand_dims((tf.range(time_step_batch_size) + \
        #                                          self._private_index) % self._buffer_max_size,
        #                                         axis=1)
        #         self._private_index.assign_add(time_step_batch_size)
        #         self._private_observations.scatter_nd_update(indices_tensor, parking_obs)
        #         self._private_actions.scatter_nd_update(indices_tensor, step.action)
        #         load = self._get_load(tf.squeeze(step.action),
        #                               tf.squeeze(augmented_obs),
        #                               True)
        #         return step.replace(action=load)
        #         # load = tf.reshape(load, [1, -1])
        #         # return step.replace(action=load)
        #
        #     return wrapped_action_collect if collect else wrapped_action_eval

        # @tf.function
        # @tf.function(jit_compile=config.USE_JIT)
        def _preprocess_sequence(self, experience: trajectory.Trajectory):
            """
            Trajectories from the buffer contain just the market environment data
            (prices), also the reward and action of every agent. Augments observation
            with parking state, also, trims the unnecessary rewards and actions and
            removes the policy info (which is used to fetch the parking state from
            the agents field _private_observations)
            """
            # print('Tracing preprocess_sequence')
            # parking_obs = self._private_observations.gather_nd(experience.policy_info)
            parking_obs = tf.gather(self._private_observations, tf.squeeze(experience.policy_info))
            # actions = self._private_actions.gather_nd(experience.policy_info)
            actions = tf.gather(self._private_actions, tf.squeeze(experience.policy_info))
            augmented_obs = tf.concat(
                (experience.observation,
                 parking_obs),
                axis=-1)
            agent_reward = experience.reward[..., self._agent_id - 1]
            # agent_action = experience.action[..., self._agent_id - 1:self._agent_id]
            # return experience.replace(observation=augmented_obs,
            return experience.replace(observation=tf.cast(augmented_obs, tf.float32),
                                      policy_info=(),
                                      reward=agent_reward,
                                      action=actions, )

        @tf.function(jit_compile=True)
        def _calculate_vehicle_distribution(self, train: bool):
            # print('Tracing calculate_vehicle_distribution')
            if train:
                departure_tensor = tf.cast(self._train_parking._vehicles[..., 2], tf.float32)
                capacity = self._capacity_train_garage
            else:
                departure_tensor = tf.cast(self._eval_parking._vehicles[..., 2], tf.float32)
                capacity = self._capacity_eval_garage
            fn = lambda t: tf.reduce_sum(tf.clip_by_value(departure_tensor - t, 0.0, 1.0))
            departure_distribution_tensor = tf.vectorized_map(fn, tf.range(0.0, 12.0))
            return departure_distribution_tensor / capacity

        @tf.function(jit_compile=True)
        def _get_parking_observation(self, train: bool):
            # print('Tracing get_parking_observation')
            if train:
                parking = self._train_parking.return_fields()
                capacity = self._capacity_train_garage
            else:
                parking = self._eval_parking.return_fields()
                capacity = self._capacity_eval_garage
            # capacity = tf.cast(capacity, tf.float16)
            next_max_charge = parking.next_max_charge
            next_min_charge = parking.next_min_charge
            next_max_discharge = parking.next_max_discharge
            next_min_discharge = parking.next_min_discharge
            max_charging_rate = parking.max_charging_rate
            max_discharging_rate = parking.max_discharging_rate
            max_acceptable = next_max_charge - next_min_discharge
            max_sign = tf.sign(max_acceptable)
            min_acceptable = next_max_discharge - next_min_charge
            min_sign = tf.sign(min_acceptable)
            max_acceptable_coefficient = tf.math.divide_no_nan(max_acceptable,
                                                               # tf.where(tf.less(0.0, max_acceptable),
                                                               #          next_max_charge,
                                                               #          next_max_discharge),
                                                               tf.maximum(max_sign * next_max_charge,
                                                                          tf.negative(max_sign) * next_max_discharge)
                                                               )
            min_acceptable_coefficient = tf.math.divide_no_nan(min_acceptable,
                                                               # tf.where(tf.less(0.0, min_acceptable),
                                                               #          next_max_discharge,
                                                               #          next_max_charge))
                                                               tf.maximum(min_sign * next_max_discharge,
                                                                          tf.negative(min_sign) * next_max_charge)
                                                               )
            # max_acceptable_sign = tf.math.sign(max_acceptable)
            # min_acceptable_sign = tf.math.sign(min_acceptable)
            # max_acceptable_coefficient = tf.math.divide_no_nan(max_acceptable,
            #                                                    tf.math.maximum(max_acceptable_sign * \
            #                                                                    next_max_charge,
            #                                                                    max_acceptable_sign * \
            #                                                                    next_max_discharge * \
            #                                                                    (-1.0)))
            # min_acceptable_coefficient = tf.math.divide_no_nan(min_acceptable,
            #                                                    tf.math.maximum(min_acceptable_sign * \
            #                                                                    next_max_charge * \
            #                                                                    (-1.0),
            #                                                                    min_acceptable_sign * \
            #                                                                    next_max_discharge))
            temp_diff = next_min_charge - next_min_discharge
            temp_sign = tf.sign(temp_diff)
            threshold_coefficient = tf.math.divide_no_nan(temp_diff,
                                                          # tf.where(tf.less(0.0, temp_diff),
                                                          #          next_max_charge,
                                                          #          next_max_discharge))
                                                          tf.maximum(temp_sign * next_max_charge,
                                                                     tf.negative(temp_sign) * next_max_discharge)
                                                          )
            # temp_diff_sign = tf.math.sign(temp_diff)
            # threshold_coefficient = tf.math.divide_no_nan(temp_diff,
            #                                               tf.math.maximum(temp_diff_sign * \
            #                                                               next_max_charge,
            #                                                               temp_diff_sign * \
            #                                                               next_max_discharge * \
            #                                                               (-1.0)))
            return tf.stack([max_acceptable_coefficient,
                             threshold_coefficient,
                             tf.negative(min_acceptable_coefficient),
                             *tf.unstack(self._calculate_vehicle_distribution(train)),
                             next_max_charge / max_charging_rate / capacity,
                             next_min_charge / max_charging_rate / capacity,
                             next_max_discharge / max_discharging_rate / capacity,
                             next_min_discharge / max_discharging_rate / capacity,
                             parking.charge_mean_priority,
                             parking.discharge_mean_priority, ],
                            axis=0), parking

        @tf.function(jit_compile=True)
        def _get_load(self, action_step: tf.Tensor, observation: tf.Tensor, parking_fields, collect_mode=True):
            # print('Tracing get_load')
            parking = self._train_parking if collect_mode else self._eval_parking
            # parking_fields = parking.return_fields()
            # observation = tf.cast(observation, tf.float16)
            length = tf.constant((self._num_of_actions - 1), dtype=tf.float32)
            start = self._num_of_multi_agent_observation_elements
            max_coefficient, threshold_coefficient, min_coefficient = tf.unstack(observation[..., start:start + 3])
            # max_coefficient = observation[..., 13]
            # threshold_coefficient = observation[..., 14]
            # min_coefficient =observation[..., 15]
            step = (max_coefficient - min_coefficient) / length
            charging_coefficient = tf.cast(action_step, dtype=tf.float32) * step + min_coefficient
            sign_coefficient = tf.sign(charging_coefficient)
            new_energy = tf.maximum(charging_coefficient * parking_fields.next_max_charge,
                                    tf.negative(
                                        charging_coefficient) * parking_fields.next_max_discharge) * sign_coefficient
            self._update_parking(parking,
                                 parking_fields,
                                 new_energy,
                                 charging_coefficient,
                                 threshold_coefficient)
            return new_energy

        @tf.function(jit_compile=True)
        def _update_parking(self,
                            parking: Parking,
                            parking_fields,
                            new_energy,
                            charging_coefficient,
                            threshold_coefficient):
            # print('Tracing update_parking')
            # parking = self._train_parking.return_fields() if train else self._eval_parking.return_fields()
            available_energy = new_energy + parking_fields.next_min_discharge - parking_fields.next_min_charge
            max_non_emergency_charge = parking_fields.next_max_charge - parking_fields.next_min_charge
            max_non_emergency_discharge = parking_fields.next_max_discharge - parking_fields.next_min_discharge
            is_charging = tf.math.sign(charging_coefficient - threshold_coefficient)
            update_coefficient = tf.clip_by_value(tf.math.sign(tf.math.abs(charging_coefficient - \
                                                                           threshold_coefficient) - \
                                                               0.02),
                                                  0.0,
                                                  1.0) * \
                                 tf.math.divide_no_nan(available_energy,
                                                       tf.math.maximum(max_non_emergency_charge * \
                                                                       is_charging,
                                                                       tf.math.negative(max_non_emergency_discharge) * \
                                                                       is_charging))
            # update_coefficient = my_round_16(update_coefficient, 2)
            parking.update(update_coefficient,
                           parking_fields.next_max_charge - parking_fields.next_min_charge,
                           parking_fields.next_max_discharge - parking_fields.next_min_discharge,
                           False)

        def wrap_external_policy_action(self, action, collect: bool):
            return self._action_wrapper(action, collect)

    return SingleAgent(list(vehicle_distribution),
                       coefficient_function,
                       ckpt_dir,
                       buffer_max_size,
                       num_of_actions,
                       capacity_train_garage,
                       capacity_eval_garage,
                       name,
                       *args,
                       **kwargs)
