import json
import math
import random
from typing import Callable, Optional, List

import tensorflow as tf
import numpy as np
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types
from tf_agents.agents import TFAgent, data_converter
from tf_agents.utils import nest_utils, common
from tf_agents.utils.common import Checkpointer
from tf_agents.utils.nest_utils import assert_matching_dtypes_and_inner_shapes

from app.models.tf_utils import my_round
from app.error_handling import ParkingIsFull
from app.models.tf_parking_4 import Parking
from app.models.tf_vehicle_4 import Vehicle, VehicleFields
from app.utils import tf_vehicle_generator, vehicle_arrival_generator_for_tf
from app.utils import generate_vehicles

from config import MAX_BUFFER_SIZE, VEHICLE_BATTERY_CAPACITY

def create_single_agent(cls: type,
                        vehicle_distribution: List,
                        name: str,
                        num_of_agents: int,
                        ckpt_dir: str,
                        capacity_train_garage: int = 100,
                        capacity_eval_garage: int = 200,
                        coefficient_function: Callable = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5,
                        *args,
                        **kwargs,):

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
                     capacity_train_garage: int = 100,
                     capacity_eval_garage: int = 200,
                     name: str = "DefaultAgent",
                     *args,
                     **kwargs):
            self._agent_id = int("".join(item for item in list(filter(str.isdigit, name))))
            if not self._agent_id:
                raise Exception('Agent name must have an integer'
                                ' to denote the agents unique id.'
                                'For example: Agent_1')
            #Parking fields. Can be removed in a different setting
            self._capacity_train_garage = capacity_train_garage
            self._capacity_eval_garage = capacity_eval_garage
            self._train_parking = Parking(self._capacity_train_garage, 'train')
            self._eval_parking = Parking(self._capacity_eval_garage, 'eval')
            self._generate_vehicles_train = generate_vehicles(coefficient_function)
            with open('data/vehicles.json') as file:
                vehicles = list(json.load(file))
            self._eval_vehicles = tf.ragged.constant(vehicles)
            # self._tf_train_vehicle_generator = iter(tf.data.Dataset.from_generator(vehicle_arrival_generator_for_tf
            #                                                                       (coefficient_function,
            #                                                                        None,
            #                                                                        True),
            #                                                                        output_signature=tf.TensorSpec
            #                                                                       (shape=(None,
            #                                                                               None),
            #                                                                        dtype=tf.float32)).prefetch(tf.data.AUTOTUNE))
            # self._tf_eval_vehicle_generator = iter(tf.data.Dataset.from_generator(vehicle_arrival_generator_for_tf
            #                                                                       (None,
            #                                                                        vehicle_distribution,
            #                                                                        False),
            #                                                                       output_signature=tf.TensorSpec
            #                                                                       (shape=(None,
            #                                                                               None),
            #                                                                        dtype=tf.float32)).prefetch(tf.data.AUTOTUNE))
            self._private_observations = tf.Variable(tf.zeros([buffer_max_size, 21]),
                                                     shape=[buffer_max_size, 21],
                                                     dtype=tf.float32,
                                                     trainable=False)
            self._private_actions = tf.Variable(tf.zeros([buffer_max_size], tf.int32),
                                                shape=[buffer_max_size],
                                                dtype=tf.int32,
                                                trainable=False)
            self._private_index = tf.Variable(0, trainable=False)
            self._buffer_max_size = buffer_max_size
            #TODO decide on weather it's needed
            #TODO remove avg vehicle list, there should not be access to global variables

            self._time_of_day = tf.Variable(0, dtype=tf.int32)
            self._eval_steps = tf.Variable(0, dtype=tf.int32)

            super(cls, self).__init__(*args, **kwargs)
            self._name = name
            self.checkpointer = Checkpointer(
                ckpt_dir='/'.join([ckpt_dir, self._name]),
                max_to_keep=3,
                agent=self,
                policy=self.policy,
                # private_observations=self._private_observations,
                private_actions=self._private_actions,
                private_index=self._private_index
            )
            self.policy.action = self._action_wrapper(self.policy.action, False)
            self.collect_policy.action = self._action_wrapper(self.collect_policy.action, True)

            buffer_time_step_spec = TimeStep(
                    step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
                    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
                    reward=tensor_spec.TensorSpec(shape=(num_of_agents,), dtype=np.float32),
                    observation=tensor_spec.BoundedTensorSpec(shape=(13,), dtype=np.float32, minimum=-1., maximum=1.),
                )
            buffer_info_spec = tensor_spec.TensorSpec(shape=(1,),
                                                      dtype=tf.int32)
            self._collect_data_context = data_converter.DataContext(
                time_step_spec=buffer_time_step_spec,
                action_spec=(),
                info_spec=buffer_info_spec,
            )

        def checkpoint_save(self):
            self.checkpointer.save(self.train_step_counter)

        @tf.function
        def reset_eval_steps(self):
            self._eval_steps.assign(0)

        @property
        def private_index(self):
            return self._private_index.value().numpy()

        @private_index.setter
        def private_index(self, index):
            self._private_index.assign(index)

        @tf.function
        def _add_new_cars(self, train_mode=True):  # TODO decide on way to choose between distributions
            #print('Tracing add_new_cars')
            if train_mode:
                vehicles = self._generate_vehicles_train(self._time_of_day)
            else:
                vehicles = tf.gather(self._eval_vehicles, self._eval_steps).to_tensor()
                vehicles = tf.cond(tf.less(0, tf.shape(vehicles)[0]),
                                   lambda: tf.concat((vehicles[..., 1:2],
                                                      vehicles[..., 2:3],
                                                      vehicles[..., 0:1]), axis=1),
                                   lambda: tf.zeros((0, 3), tf.float32)
                                  )
            max_min_charges = tf.repeat([[VEHICLE_BATTERY_CAPACITY, 0.]], repeats=tf.shape(vehicles)[0], axis=0)
            vehicles = tf.concat((vehicles, max_min_charges), axis=1)
            vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
            if train_mode:
                self._train_parking.assign_vehicles(vehicles)
            else:
                self._eval_parking.assign_vehicles(vehicles)



        @tf.function
        def _train(self, experience: types.NestedTensor,
                   weights: types.Tensor) -> LossInfo:
            #print('Tracing train')
            return super()._train(self.preprocess_sequence(experience), weights)

        @tf.function
        def _get_load(self, action_step: tf.Tensor, observation: tf.Tensor, collect_mode = True):
            #print("Tracing get_load")
            parking = self._train_parking.return_fields() if collect_mode else self._eval_parking.return_fields()
            length = tf.constant((21.0), dtype=tf.float32)
            max_coefficient = observation[..., 13]
            threshold_coefficient = observation[..., 14]
            min_coefficient =observation[..., 15]
            step = (max_coefficient - min_coefficient) / (length-1.0)
            charging_coefficient = tf.cast(action_step, dtype=tf.float32) * step + min_coefficient
            charging_coefficient = my_round(charging_coefficient, 4)
            is_charging = tf.less_equal(threshold_coefficient, charging_coefficient)
            is_buying = tf.less_equal(tf.constant(0.0), charging_coefficient)
            max_energy = parking.next_max_charge if is_buying else parking.next_max_discharge
            new_energy = my_round(max_energy * charging_coefficient, 2)

            self._update_parking(collect_mode,
                                 is_charging,
                                 new_energy,
                                 charging_coefficient,
                                 threshold_coefficient)
            return new_energy

        def _action_wrapper(self, action, collect=True):
            """
            Wraps the action method of a policy to allow it to consume
            timesteps from the single buffer, augmenting the observation
            with the saved garage state (or not if it's called during training)
            """

            @tf.function
            def wrapped_action_eval(time_step: TimeStep,
                                    policy_state: types.NestedTensor = (),
                                    seed: Optional[types.Seed] = None,) -> PolicyStep:
                #print('Tracing wrapped_action_eval')
                try:
                    time_step = nest_utils.prune_extra_keys(self.policy.time_step_spec, time_step)
                    policy_state = nest_utils.prune_extra_keys(
                        self.policy.policy_state_spec, policy_state)
                    nest_utils.assert_same_structure(
                        time_step,
                        self.policy.time_step_spec,
                        message='time_step and time_step_spec structures do not match')
                    return action(time_step,
                                  policy_state,
                                  seed)
                except (TypeError, ValueError):
                    if ~time_step.is_last():
                        self._add_new_cars(False)
                        self._eval_steps.assign_add(1)
                    parking_obs = tf.reshape(self._get_parking_observation(False),
                                             [-1,21])
                    augmented_obs = tf.concat((time_step.observation,
                                               parking_obs),
                                              -1)
                    reward = time_step.reward[..., self._agent_id - 1]
                    new_time_step = time_step._replace(observation=augmented_obs,
                                                       reward=reward)
                    step = action(
                        new_time_step,
                        policy_state,
                        seed,)
                    load = self._get_load(tf.squeeze(step.action),
                                          tf.squeeze(augmented_obs),
                                          False)
                    load = tf.reshape(load, [1, -1])
                    return step.replace(action=load)

            @tf.function
            def wrapped_action_collect(time_step: TimeStep,
                               policy_state: types.NestedTensor = (),
                               seed: Optional[types.Seed] = None,) -> PolicyStep:
                #print('Tracing wrapped_action_collect')
                if ~time_step.is_last():
                    self._add_new_cars(True)
                    self._time_of_day.assign_add(1)
                else:
                    self._time_of_day.assign(0)
                parking_obs = tf.reshape(self._get_parking_observation(True),
                                         [-1,21])
                augmented_obs = tf.concat(
                    (time_step.observation,
                    parking_obs),
                    -1,)
                reward = time_step.reward[..., self._agent_id - 1]
                new_time_step = time_step._replace(observation=augmented_obs,
                                                   reward=reward)
                step = action(new_time_step,
                              policy_state,
                              seed,)
                time_step_batch_size = tf.shape(time_step.observation)[0]
                indices_tensor = tf.reshape((tf.range(time_step_batch_size) +
                                             self._private_index) %
                                            self._buffer_max_size,
                                            [-1, 1])
                self._private_index.assign_add(time_step_batch_size)
                self._private_observations.scatter_nd_update(indices_tensor, parking_obs)
                self._private_actions.scatter_nd_update(indices_tensor, step.action)
                load = self._get_load(tf.squeeze(step.action),
                                      tf.squeeze(augmented_obs),
                                      True)
                load = tf.reshape(load, [1, -1])
                return step.replace(action=load)

            return wrapped_action_collect if collect else wrapped_action_eval

        @tf.function
        def _preprocess_sequence(self, experience: trajectory.Trajectory):
            """
            Trajectories from the buffer contain just the market environment data
            (prices), also the reward and action of every agent. Augments observation
            with parking state, also, trims the unnecessary rewards and actions and
            removes the policy info (which is used to fetch the parking state from
            the agents field _private_observations)
            """
            #print("Tracing preprocess_sequence")
            parking_obs = self._private_observations.gather_nd(experience.policy_info)
            actions = self._private_actions.gather_nd(experience.policy_info)
            augmented_obs = tf.concat(
                (experience.observation,
                 parking_obs),
                axis=-1)
            agent_reward = experience.reward[..., self._agent_id - 1]
            # agent_action = experience.action[..., self._agent_id - 1:self._agent_id]
            return experience.replace(observation=augmented_obs,
                                      policy_info=(),
                                      reward=agent_reward,
                                      action=actions,)

        @tf.function
        def _calculate_vehicle_distribution(self, train: tf.Tensor):
            #print("Tracing calculate_vehicle_distribution")
            departure_tensor = tf.cond(tf.constant(train),
                                       lambda: self._train_parking.vehicles,
                                       lambda: self._eval_parking.vehicles)
            capacity = tf.cond(tf.constant(train),
                               lambda: self._capacity_train_garage,
                               lambda: self._capacity_eval_garage)
            departure_tensor = tf.cast(tf.reshape(departure_tensor[..., 2], [-1]), tf.int32)
            y, idx, count = tf.unique_with_counts(departure_tensor)
            departure_count_tensor = tf.tensor_scatter_nd_update(tf.zeros((12,), tf.int32),
                                                                 tf.reshape(y - 1, [-1, 1]),
                                                                 count,)
            departure_distribution_tensor = tf.math.cumsum(departure_count_tensor,
                                                           reverse=True)
            return tf.cast(departure_distribution_tensor / capacity, tf.float32)

        @tf.function
        def _get_parking_observation(self, train: tf.Tensor):
            #print('Tracing get_parking_observation')
            parking = tf.cond(tf.constant(train),
                              self._train_parking.return_fields,
                              self._eval_parking.return_fields)
            capacity = tf.cast(tf.cond(tf.constant(train),
                                       lambda: self._capacity_train_garage,
                                       lambda: self._capacity_eval_garage),
                               dtype=tf.float32)
            next_max_charge = parking.next_max_charge
            next_min_charge = parking.next_min_charge
            next_max_discharge = parking.next_max_discharge
            next_min_discharge = parking.next_min_discharge
            max_charging_rate = parking.max_charging_rate
            max_discharging_rate = parking.max_discharging_rate
            max_acceptable = next_max_charge - next_min_discharge
            min_acceptable = next_max_discharge - next_min_charge
            max_acceptable_coefficient = tf.cond(tf.not_equal(max_acceptable, 0.0),
                                                 lambda: max_acceptable / tf.cond(tf.less(0.0, max_acceptable),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)
            min_acceptable_coefficient = tf.cond(tf.not_equal(min_acceptable, 0.0),
                                                 lambda: min_acceptable / tf.cond(tf.less(min_acceptable, 0.0),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)

            temp_diff = next_min_charge - next_min_discharge
            threshold_coefficient = tf.cond(tf.not_equal(temp_diff, 0.0),
                                                 lambda: temp_diff / tf.cond(tf.less(0.0, temp_diff),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)
            return tf.stack([max_acceptable_coefficient,
                             threshold_coefficient,
                             -min_acceptable_coefficient,
                             *tf.unstack(self._calculate_vehicle_distribution(train)),
                             next_max_charge / max_charging_rate / capacity,
                             next_min_charge / max_charging_rate / capacity,
                             next_max_discharge / max_discharging_rate / capacity,
                             next_min_discharge / max_discharging_rate / capacity,
                             parking.charge_mean_priority,
                             parking.discharge_mean_priority,],
                            axis = 0)

        @tf.function
        def _update_parking(self,
                            train,
                            is_charging,
                            new_energy,
                            charging_coefficient,
                            threshold_coefficient):
            #print('Tracing update_parking')
            parking = self._train_parking.return_fields() if train else self._eval_parking.return_fields()
            available_energy = new_energy + parking.next_min_discharge - parking.next_min_charge
            max_non_emergency_charge = parking.next_max_charge - parking.next_min_charge
            max_non_emergency_discharge = parking.next_max_discharge - parking.next_min_discharge
            update_coefficient = tf.cond(
                tf.less(0.02, my_round(tf.math.abs(charging_coefficient - threshold_coefficient), 2)),
                lambda: available_energy / tf.cond(is_charging,
                                                   lambda: max_non_emergency_charge,
                                                   lambda: max_non_emergency_discharge),
                lambda: tf.constant(0.0)
            )
            update_coefficient = my_round(update_coefficient, 2)
            if train:
                self._train_parking.update(update_coefficient)
            else:
                self._eval_parking.update(update_coefficient)

    return SingleAgent(list(vehicle_distribution),
                       coefficient_function,
                       ckpt_dir,
                       MAX_BUFFER_SIZE,
                       capacity_train_garage,
                       capacity_eval_garage,
                       name,
                       *args,
                       **kwargs)
