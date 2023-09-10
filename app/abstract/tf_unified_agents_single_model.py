from typing import List

import tensorflow as tf
from tf_agents.typing.types import SpecTensorOrArray

from app.models.tf_parking_single_agent import Parking
from app.utils import generate_vehicles_single_model
from config import VEHICLE_BATTERY_CAPACITY


class UnifiedAgent(tf.Module):
    def __init__(
            self,
            single_agent_list: List,
            name: str = "Unified_Agent"
    ):
        # Parking fields. Can be removed in a different setting

        self._capacity_train_garage = single_agent_list[0]._capacity_train_garage
        self._capacity_eval_garage = single_agent_list[0]._capacity_eval_garage
        if any([(agent._capacity_train_garage, agent._capacity_eval_garage) !=
                (self._capacity_train_garage, self._capacity_eval_garage) for agent in
                single_agent_list]):
            raise Exception('All agents must have equal garage capacities')
        self._num_of_agents = len(single_agent_list)
        self._num_of_actions = tf.expand_dims(
            tf.constant([agent._num_of_actions for agent in single_agent_list], tf.float32), axis=-1)
        self._train_parking = Parking(self._capacity_train_garage, self._num_of_agents, 'train')
        self._eval_parking = Parking(self._capacity_eval_garage, self._num_of_agents, 'eval')
        self._grouped_agent_list = []
        grouped_agent_numbers = []
        grouped_coefficient_function_list = []
        while single_agent_list:
            temp_agent = single_agent_list.pop(0)
            self._grouped_agent_list.append(temp_agent)
            i = 1
            temp_fun = temp_agent.coefficient_function
            for agent in single_agent_list[:]:
                if agent.coefficient_function == temp_fun:
                    self._grouped_agent_list.append(agent)
                    single_agent_list.remove(agent)
                    i += 1
            grouped_agent_numbers.append(i)
            grouped_coefficient_function_list.append(temp_fun)
        self._generate_vehicles_train = generate_vehicles_single_model(
            coefficient_function_list=grouped_coefficient_function_list,
            num_of_agents_list=grouped_agent_numbers,
            capacity=self._capacity_train_garage
        )
        self._eval_vehicles = tf.stack([agent._eval_vehicles for agent in self._grouped_agent_list], axis=0)
        self._collect_steps = tf.Variable(-1,
                                          dtype=tf.int64,
                                          trainable=False,
                                          name=name + ': time_of_day')
        self._eval_steps = tf.Variable(-1,
                                       dtype=tf.int64,
                                       trainable=False,
                                       name=name + ': eval steps')
        self._name = name
        self._num_of_multi_agent_observation_elements = self._grouped_agent_list[0].time_step_spec.observation.shape[
                                                            0] - 21
        if any([agent.time_step_spec.observation.shape[0] - 21 != self._num_of_multi_agent_observation_elements for
                agent in
                self._grouped_agent_list]):
            raise Exception('Agents must have the same observation shape')

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

    @private_index.setter
    def private_index(self, index):
        self._private_index.assign(index)

    @property
    def grouped_agent_list(self):
        return self._grouped_agent_list

    @property
    def agent_list(self):
        return self._grouped_agent_list

    @property
    def train_vehicles_generator(self):
        return self._generate_vehicles_train

    def _add_new_cars(self, train_mode=True):  # TODO decide on way to choose between distributions
        # print('Tracing add_new_cars')
        if train_mode:
            # vehicles = self._generate_vehicles_train(self._collect_steps % tf.constant(24, tf.int64))
            max_min_charges = tf.repeat((tf.expand_dims(tf.repeat([[VEHICLE_BATTERY_CAPACITY, 0.0] + [0.0] * 8],
                                                                  self._capacity_train_garage,
                                                                  axis=0),
                                                        axis=0)),
                                        self._num_of_agents,
                                        axis=0)
            vehicles = tf.concat((self._generate_vehicles_train(self._collect_steps % tf.constant(24, tf.int64)),
                                  max_min_charges),
                                 axis=-1)
        else:
            max_min_charges = tf.repeat(tf.expand_dims(tf.repeat([[VEHICLE_BATTERY_CAPACITY, 0.0] + [0.0] * 8],
                                                                 self._capacity_eval_garage,
                                                                 axis=0),
                                                       axis=0),
                                        self._num_of_agents,
                                        axis=0)
            vehicles = tf.concat((self._eval_vehicles[:, self._eval_steps],
                                  max_min_charges),
                                 axis=-1)
        if train_mode:
            self._train_parking.assign_vehicles(vehicles)
        else:
            self._eval_parking.assign_vehicles(vehicles)

    def get_action(self, step: SpecTensorOrArray, augmented_obs: SpecTensorOrArray, parking_fields, collect=True):
        load = self._get_load(step,
                              augmented_obs,
                              parking_fields,
                              collect)
        return tf.squeeze(load)

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
        return tf.concat((tf.repeat([observation], self._num_of_agents, axis=0), parking_obs), axis=-1), parking_fields

    @tf.function(jit_compile=True)
    def _calculate_vehicle_distribution(self, train: bool):
        # print('Tracing calculate_vehicle_distribution')
        if train:
            departure_tensor = tf.cast(self._train_parking._vehicles[..., 2:3], tf.float32)
            capacity = self._capacity_train_garage
        else:
            departure_tensor = tf.cast(self._eval_parking._vehicles[..., 2:3], tf.float32)
            capacity = self._capacity_eval_garage
        fn = lambda t: tf.reduce_sum(tf.clip_by_value(departure_tensor - t, 0.0, 1.0), axis=-2)
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
        temp_diff = next_min_charge - next_min_discharge
        temp_sign = tf.sign(temp_diff)
        threshold_coefficient = tf.math.divide_no_nan(temp_diff,
                                                      tf.maximum(temp_sign * next_max_charge,
                                                                 tf.negative(temp_sign) * next_max_discharge)
                                                      )
        return tf.concat([max_acceptable_coefficient,
                          threshold_coefficient,
                          tf.negative(min_acceptable_coefficient),
                          *tf.unstack(self._calculate_vehicle_distribution(train)),
                          next_max_charge / max_charging_rate / capacity,
                          next_min_charge / max_charging_rate / capacity,
                          next_max_discharge / max_discharging_rate / capacity,
                          next_min_discharge / max_discharging_rate / capacity,
                          parking.charge_mean_priority,
                          parking.discharge_mean_priority, ],
                         axis=-1), parking

    @tf.function(jit_compile=True)
    def _get_load(self, action_step: tf.Tensor, observation: tf.Tensor, parking_fields, collect_mode=True):
        # print('Tracing get_load')
        parking = self._train_parking if collect_mode else self._eval_parking
        # parking_fields = parking.return_fields()
        # observation = tf.cast(observation, tf.float16)
        # length = tf.constant((self._num_of_actions - 1), dtype=tf.float32)
        length = self._num_of_actions - 1.0
        start = self._num_of_multi_agent_observation_elements
        max_coefficient, threshold_coefficient, min_coefficient = tf.unstack(
            tf.expand_dims(observation[..., start:start + 3], axis=-1), axis=-2)
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
