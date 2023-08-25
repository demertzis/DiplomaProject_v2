import json
from typing import Any, Dict
import config as cfg
import app.models.tf_vehicle_9 as Vehicle
# from app.models.tf_utils import my_round

import tensorflow as tf

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
    def __init__(self, capacity: int, name: str):
        self._next_max_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._next_min_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._next_max_discharge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._next_min_discharge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._charge_mean_priority = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._discharge_mean_priority = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._current_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        # self._max_charging_rate = tf.constant(cfg.MAX_CHARGING_RATE, tf.float32).numpy()
        self._max_charging_rate = float(cfg.MAX_CHARGING_RATE)
        # self._max_discharging_rate = tf.constant(cfg.MAX_DISCHARGING_RATE, tf.float32).numpy()
        self._max_discharging_rate = float(cfg.MAX_DISCHARGING_RATE)

        self._capacity = tf.constant(capacity, tf.int64)
        self._capacity_int32 = capacity
        self._vehicles = tf.Variable(tf.zeros(shape=(capacity, 13), dtype=tf.float32), trainable=False)
        self._num_of_vehicles = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.name = name,
        self._max_charging_rate_32 = tf.cast(self._max_charging_rate, tf.float32)
        self._max_discharging_rate_32 = tf.cast(self._max_discharging_rate, tf.float32)

    @property
    def vehicles(self):
        return self._vehicles.value()[:self._num_of_vehicles.value()]

    def return_fields(self):
        return ParkingFields(self._next_max_charge,
                             self._next_min_charge,
                             self._next_max_discharge,
                             self._next_min_discharge,
                             self._charge_mean_priority,
                             self._discharge_mean_priority,
                             self._current_charge,
                             self._max_charging_rate_32,
                             self._max_discharging_rate_32)

    def get_current_vehicles(self):
        """
        Get current amount of vehicles in parking
        """
        return len(self._vehicles)

    def get_capacity(self):
        """
        Get capacity of parking

        ### Returns
            capacity (`int`): The total parking spaces
        """
        return self._capacity

    def get_current_energy(self):
        """
        Get total energy stored in the parking

        ### Returns:
            float : The sum of all vehicles' current energy
        """
        return self._current_charge

    def get_next_max_charge(self):
        """
        Get next cumulative maximum charge

        ### Returns:
            float : The sum of all vehicles next max charge
        """
        return self._next_max_charge

    def get_next_min_charge(self):
        """
        Get next cumulative minimum charge

        ### Returns:
            float : The sum of all vehicles next min charge
        """
        return self._next_min_charge

    def get_next_max_discharge(self):
        """
        Get next cumulative maximum discharge

        ### Returns:
            float : The sum of all vehicles next max discharge
        """
        return self._next_max_discharge

    def get_next_min_discharge(self):
        """
        Get next cumulative minimum discharge

        ### Returns:
            float : The sum of all vehicles next min discharge
        """
        return self._next_min_discharge

    def get_charge_mean_priority(self):
        """
        Get mean charge priority
        ### Returns
            float : The mean charge priority
        """
        return self._charge_mean_priority

    def get_discharge_mean_priority(self):
        """
        Get mean discharge priority
        ### Returns
            float : The mean discharge priority
        """
        return self._discharge_mean_priority

    def get_max_charging_rate(self):
        """
        Get max charging rate
        ### Returns
            float : The max charging rate
        """
        return self._max_charging_rate

    def get_max_discharging_rate(self):
        """
        Get max discharging rate
        ### Returns
            float : The max discharging rate
        """
        return self._max_discharging_rate

    def assign_vehicles(self, vehicles: tf.Tensor):
        """
        Assign vehicle to a parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        # print('Tracing assign_vehicle')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicles_added_mult = tf.roll(tf.concat(((tf.zeros([self._capacity, 1], tf.float32)),
                                                  tf.ones([self._capacity, 1], tf.float32)),
                                                axis=0),
                                      shift=num_of_vehicles,
                                      axis=0)[self._capacity:]
        rolled_vehicles = tf.roll(vehicles, shift=num_of_vehicles, axis=0)
        vehicles_added_final = rolled_vehicles * vehicles_added_mult
        self._num_of_vehicles.assign_add(tf.cast(tf.reduce_sum(tf.clip_by_value(vehicles_added_final[..., 2], 0.0, 1.0)),
                                                 tf.int64))
        new_charges = tf.reduce_sum(vehicles_added_final[..., 0])
        self._current_charge.assign_add(new_charges)
        parked_tensor = Vehicle.park(vehicles_added_final,
                                     self._max_charging_rate,
                                     self._max_discharging_rate,
                                     self._capacity_int32)
        self._vehicles.assign_add(parked_tensor)
        self._update_parking_state(self._vehicles)

    # @tf.function
    def _update_parking_state(self, vehicles):
        # print('Tracing update_parking_state')
        num_of_vehicles = tf.cast(self._num_of_vehicles, tf.float32)
        priorities = vehicles[..., 5:7]
        current_charges = vehicles[..., 0]
        charges = vehicles[..., 9:]
        charges_sum = tf.reduce_sum(charges, axis=0)
        priorities_sum = tf.reduce_sum(priorities, axis=0)
        priorities_mean = tf.math.divide_no_nan(priorities_sum, num_of_vehicles)
        self._current_charge.assign(tf.reduce_sum(current_charges))
        self._next_max_charge.assign(charges_sum[0])
        self._next_min_charge.assign(charges_sum[1])
        self._next_max_discharge.assign(charges_sum[2])
        self._next_min_discharge.assign(charges_sum[3])
        self._charge_mean_priority.assign(priorities_mean[0])
        self._discharge_mean_priority.assign(priorities_mean[1])

    @tf.function(jit_compile=True)
    def update(self, charging_coefficient):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        #print('Tracing update (parking)')
        new_next_max_charge = self._next_max_charge - self._next_min_charge
        new_next_max_discharge = self._next_max_discharge - self._next_min_discharge
        staying_vehicles = _update(self._vehicles,
                                   charging_coefficient,
                                   self._capacity_int32,
                                   new_next_max_charge,
                                   new_next_max_discharge)
        self._next_max_charge.assign(new_next_max_charge)
        self._next_max_discharge.assign(new_next_max_discharge)
        self._update_parking_state(staying_vehicles)
        self._num_of_vehicles.assign(tf.cast(tf.reduce_sum(tf.clip_by_value(staying_vehicles[..., 2], 0.0, 1.0)),
                                             tf.int64))
        self._vehicles.assign(staying_vehicles)


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


# @tf.function
def _depart_vehicles(vehicle_array: tf.Tensor):
    """
    Filter out all vehicles that have left the parking
    """
    #print('Tracing depart_vehicles')
    sorted_args = tf.argsort(vehicle_array[..., 2], direction='DESCENDING')
    vehicle_list = tf.gather(vehicle_array, sorted_args)
    mult_tensor = tf.clip_by_value(vehicle_list[..., 2:3], 0.0, 1.0)
    return vehicle_list * mult_tensor

def _update_energy_state(charging_coefficient, vehicles, capacity, next_max_charge, next_max_discharge):
    """
    Update energy state of parking

    ### Arguments:
        charging_coefficient (``float``) :
            description: The ratio of the used charging/discharging capacity
    """
    #print('Tracing update_energy_state')
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
                                                 capacity)
    return new_vehicles

@tf.function(jit_compile=True)
def _update(vehicles, charging_coefficient, capacity, next_max_charge, next_max_discharge):
    """
    Given the action input, it performs an update step

    ### Arguments:
        charging_coefficient (``float``) :
            description: The charging coefficient
    """
    print('Tracing _update (parking)')
    new_vehicles = _update_energy_state(charging_coefficient, vehicles, capacity, next_max_charge, next_max_discharge)
    staying_vehicles = _depart_vehicles(new_vehicles)
    return staying_vehicles
