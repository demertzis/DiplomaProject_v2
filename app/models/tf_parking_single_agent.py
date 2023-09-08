import json
from typing import Any, Dict

import tensorflow as tf

import app.models.tf_vehicle_single_agent as Vehicle
import config as cfg


# from app.models.tf_utils import my_round


class ParkingFields(tf.experimental.ExtensionType):
    _next_max_charge: tf.Tensor
    _next_min_charge: tf.Tensor
    _next_max_discharge: tf.Tensor
    _next_min_discharge: tf.Tensor
    _charge_mean_priority: tf.Tensor
    _discharge_mean_priority: tf.Tensor
    _current_charge: tf.Tensor
    _max_charging_rate: tf.Tensor
    _max_discharging_rate: tf.Tensor

    @property
    def next_max_charge(self):
        return self._next_max_charge

    @property
    def next_min_charge(self):
        return self._next_min_charge

    @property
    def next_max_discharge(self):
        return self._next_max_discharge

    @property
    def next_min_discharge(self):
        return self._next_min_discharge

    @property
    def charge_mean_priority(self):
        return self._charge_mean_priority

    @property
    def discharge_mean_priority(self):
        return self._discharge_mean_priority

    @property
    def current_charge(self):
        return self._current_charge

    @property
    def max_charging_rate(self):
        return self._max_charging_rate

    @property
    def max_discharging_rate(self):
        return self._max_discharging_rate


class Parking:
    """
    A class representing a V2G parking facility

    ### Arguments:
        capacity (``int``) :
            description: The total parking spaces

    ### Attributes
        parking_spaces (``ParkingSpace[]``) :
            description: An array containing all available parking spaces objects
    """

    def __init__(self, capacity: int, num_of_agents: int, name: str):
        # self._max_charging_rate = tf.constant(cfg.MAX_CHARGING_RATE, tf.float32).numpy()
        self._max_charging_rate = float(cfg.MAX_CHARGING_RATE)
        # self._max_discharging_rate = tf.constant(cfg.MAX_DISCHARGING_RATE, tf.float32).numpy()
        self._max_discharging_rate = float(cfg.MAX_DISCHARGING_RATE)

        self._capacity = tf.constant(capacity, tf.int64)
        self._num_of_agents = num_of_agents
        self._capacity_int32 = capacity
        self._vehicles = tf.Variable(tf.zeros(shape=(num_of_agents, capacity, 13), dtype=tf.float32), trainable=False)
        self._num_of_vehicles = tf.Variable(tf.zeros(shape=[num_of_agents], dtype=tf.int64), trainable=False)
        self.name = name,
        self._max_charging_rate_32 = tf.cast(self._max_charging_rate, tf.float32)
        self._max_discharging_rate_32 = tf.cast(self._max_discharging_rate, tf.float32)

    @property
    def vehicles(self):
        return self._vehicles.value()[:self._num_of_vehicles.value()]

    def return_fields(self):
        return ParkingFields(*self._update_parking_state(self._vehicles))

    def assign_vehicles(self, vehicles: tf.Tensor):
        """
        Assign vehicles to the parking space

        ### Arguments:
            A vehicles tensor of shape [capacity, 13]
        """
        # print('Tracing assign_vehicle')
        num_of_vehicles = self._num_of_vehicles.value()
        unrolled_tensor = tf.concat(((tf.zeros([self._capacity, 1], tf.float32)),
                                     tf.ones([self._capacity, 1], tf.float32)),
                                    axis=0)

        vehicles_added_mult = tf.vectorized_map(lambda t: tf.roll(unrolled_tensor, t, axis=0)[self._capacity:],
                                                num_of_vehicles)
        # rolled_vehicles = tf.vectorized_map(lambda t: ,
        #                                     tf.concat(tf.expand_dims(tf.cast(vehicles, tf.float32), axis=1),
        #                                               vehicles),
        #                                     axis)
        rolled_vehicles = tf.vectorized_map(lambda t: tf.roll(t[1], t[0], axis=0),
                                            [num_of_vehicles, vehicles])
        vehicles_added_final = rolled_vehicles * vehicles_added_mult
        self._num_of_vehicles.assign_add(tf.cast(tf.reduce_sum(tf.clip_by_value(vehicles_added_final[..., 2], 0.0, 1.0),
                                                               axis=-1),
                                                 tf.int64))
        parked_tensor = Vehicle.park(vehicles_added_final,
                                     self._max_charging_rate,
                                     self._max_discharging_rate,
                                     self._capacity_int32,
                                     self._num_of_agents)
        self._vehicles.assign_add(parked_tensor)

    # @tf.function
    def _update_parking_state(self, vehicles):
        # print('Tracing update_parking_state')
        num_of_vehicles = tf.cast(self._num_of_vehicles, tf.float32)
        priorities = vehicles[..., 5:7]
        current_charges = vehicles[..., 0]
        charges = vehicles[..., 9:]
        charges_sum = tf.reduce_sum(charges, axis=-2)
        priorities_sum = tf.reduce_sum(priorities, axis=-2)
        priorities_mean = tf.math.divide_no_nan(priorities_sum, tf.expand_dims(num_of_vehicles, axis=-1))
        return [charges_sum[..., 0],
                charges_sum[..., 1],
                charges_sum[..., 2],
                charges_sum[..., 3],
                priorities_mean[..., 0],
                priorities_mean[..., 1],
                tf.reduce_sum(current_charges, axis=-1),
                self._max_charging_rate_32,
                self._max_discharging_rate]

    @tf.function(jit_compile=True)
    def update(self, charging_coefficient, new_next_max_charge, new_next_max_discharge, compute_charges=True):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        # print('Tracing update (parking)')
        if compute_charges:
            fields = self.return_fields()
            new_next_max_charge = fields.next_max_charge - fields.next_min_charge
            new_next_max_discharge = fields.next_max_discharge - fields.next_min_discharge
        staying_vehicles = self._update(self._vehicles,
                                        charging_coefficient,
                                        new_next_max_charge,
                                        new_next_max_discharge)
        self._num_of_vehicles.assign(tf.cast(tf.reduce_sum(tf.clip_by_value(staying_vehicles[..., 2], 0.0, 1.0)),
                                             tf.int64))
        self._vehicles.assign(staying_vehicles)

    # @tf.function
    def _depart_vehicles(self, vehicle_array: tf.Tensor):
        """
        Filter out all vehicles that have left the parking
        """
        # print('Tracing depart_vehicles')
        sorted_args = tf.argsort(vehicle_array[..., 2], direction='DESCENDING')
        vehicle_list = tf.gather(vehicle_array, sorted_args)
        mult_tensor = tf.clip_by_value(vehicle_list[..., 2:3], 0.0, 1.0)
        return vehicle_list * mult_tensor

    def _update_energy_state(self, charging_coefficient, vehicles, next_max_charge, next_max_discharge):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        # print('Tracing update_energy_state')
        # is_charging = tf.less_equal(0.0, charging_coefficient)
        # neg_is_charging = tf.logical_not(is_charging)
        is_charging = tf.cast(tf.less_equal(0.0, charging_coefficient), tf.float32)
        neg_is_charging = tf.cast(tf.less(charging_coefficient, 0.0), tf.float32)
        vehicle_array = Vehicle.update_emergency_demand(vehicles)
        norm_constants = tf.math.divide_no_nan(tf.reduce_sum(vehicle_array[..., 9:12:2] * vehicle_array[..., 5:7],
                                                             axis=0),
                                               tf.stack((next_max_charge, next_max_discharge),
                                                        axis=0))
        normalization_constant = tf.maximum(is_charging * norm_constants[0], neg_is_charging * norm_constants[1])
        tensor_mapping = is_charging * vehicle_array[..., 5] + neg_is_charging * vehicle_array[..., 6]
        sorted_args = tf.argsort(tensor_mapping, direction='DESCENDING')
        sorted_vehicles = tf.gather(vehicle_array, sorted_args)
        new_vehicles = Vehicle.update_current_charge(charging_coefficient,
                                                     normalization_constant,
                                                     sorted_vehicles,
                                                     self._capacity_int32,
                                                     self._num_of_agents)
        return new_vehicles

    @tf.function(jit_compile=True)
    def _update(self, vehicles, charging_coefficient, next_max_charge, next_max_discharge):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        print('Tracing _update (parking)')
        new_vehicles = self._update_energy_state(charging_coefficient, vehicles, next_max_charge, next_max_discharge)
        staying_vehicles = self._depart_vehicles(new_vehicles)
        return staying_vehicles

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Parking.__name__,
            "_name": self.name,
            "max_charging_rage": self._max_charging_rate,
            "max_discharging_rate": self._max_discharging_rate,
            "vehicles": list(map(lambda v: v.toJson(), self._vehicles)),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
