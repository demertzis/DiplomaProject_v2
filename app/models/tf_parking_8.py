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

    _next_max_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _next_min_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _next_max_discharge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _next_min_discharge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _charge_mean_priority = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _discharge_mean_priority = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _current_charge = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _max_charging_rate = tf.constant(cfg.MAX_CHARGING_RATE, tf.float32).numpy()
    _max_discharging_rate = tf.constant(cfg.MAX_DISCHARGING_RATE, tf.float32).numpy()

    def __init__(self, capacity: int, name: str):
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

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32)])
    # def assign_vehicles(self, vehicles: tf.Tensor):
    #     """
    #     Assign vehicle to a parking space
    #
    #     ### Arguments:
    #         vehicle (``Vehicle``) :
    #             description: A non-parked instance of the vehicle class
    #
    #     ### Raises:
    #         ParkingIsFull: The parking has no free spaces
    #     """
    #     #print('Tracing assign_vehicle')
    #     num_of_vehicles = self._num_of_vehicles.value()
    #     vehicles_added_number = tf.minimum(tf.shape(vehicles, tf.int64)[0], self._capacity - num_of_vehicles)
    #     self._num_of_vehicles.assign_add(vehicles_added_number)
    #     vehicles_added = tf.gather(vehicles, tf.range(vehicles_added_number))
    #     vehicles_added_padded = tf.pad(vehicles_added,
    #                                    [[num_of_vehicles, self._capacity - num_of_vehicles - vehicles_added_number],
    #                                     [0, 0]],
    #                                    constant_values=0.0
    #                                   )
    #     new_charges = tf.cast(tf.reduce_sum(vehicles_added[..., 0]), tf.float32)
    #     self._current_charge.assign_add(new_charges)
    #     parked_tensor = Vehicle.park(vehicles_added_padded,
    #                                  self._max_charging_rate,
    #                                  self._max_discharging_rate,
    #                                  self._capacity_int32)
    #     self._vehicles.assign_add(parked_tensor)
    #     # self._update_batch_vehicles(self._vehicles)
    #     self._update_parking_state(self._vehicles)

    def assign_vehicles(self, vehicles: tf.Tensor):
        """
        Assign vehicle to a parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        #print('Tracing assign_vehicle')
        num_of_vehicles = self._num_of_vehicles.value()
        # arrived_vehicles = tf.reduce_sum(tf.where(tf.less(0.0, vehicles[..., 2]), 1, 0))
        # vehicles_added_number = tf.minimum(tf.cast(arrived_vehicles, tf.int64), self._capacity - num_of_vehicles)
        # self._num_of_vehicles.assign_add(vehicles_added_number)
        # vehicles_added_mult = tf.where(tf.less(tf.expand_dims(tf.range(self._capacity), axis=1),
        #                                        vehicles_added_number),
        #                                1.0,
        #                                0.0)
        indexes = tf.expand_dims(tf.range(self._capacity), axis=1)
        vehicles_added_mult = tf.where(tf.less_equal(num_of_vehicles, indexes),
                                         1.0,
                                         0.0)
        rolled_vehicles = tf.roll(vehicles, shift=num_of_vehicles, axis=0)
        vehicles_added_final = rolled_vehicles * vehicles_added_mult
        self._num_of_vehicles.assign_add(tf.cast(tf.reduce_sum(tf.where(tf.less(0.0,
                                                                                vehicles_added_final[..., 2]),
                                                                        1,
                                                                        0)),
                                                 tf.int64))
        new_charges = tf.reduce_sum(vehicles_added_final[..., 0])
        # vehicles_added = vehicles * vehicles_added_mult
        # vehicles_added_final = tf.roll(vehicles_added, shift=num_of_vehicles, axis=0)
        # new_charges = tf.cast(tf.reduce_sum(vehicles_added[..., 0]), tf.float32)
        self._current_charge.assign_add(new_charges)
        parked_tensor = Vehicle.park(vehicles_added_final,
                                     self._max_charging_rate,
                                     self._max_discharging_rate,
                                     self._capacity_int32)
        self._vehicles.assign_add(parked_tensor)
        # self._update_batch_vehicles(self._vehicles)
        self._update_parking_state(self._vehicles)

    # @tf.function
    def depart_vehicles(self, vehicle_array: tf.Tensor):
        """
        Filter out all vehicles that have left the parking
        """
        #print('Tracing depart_vehicles')
        # num_of_vehicles = self._num_of_vehicles.value()
        # vehicle_list = self._vehicles.value()[:num_of_vehicles]
        # vehicle_array = self._vehicles
        # vehicle_list = self._sort_vehicles_by_departure(self._vehicles)
        sorted_args = tf.argsort(vehicle_array[..., 2], direction='DESCENDING')
        vehicle_list = tf.gather(vehicle_array, sorted_args)
        # mult_tensor = tf.where(tf.less(0.0, vehicle_list[..., 2:3]), 1.0, 0.0)
        mult_tensor = tf.clip_by_value(vehicle_list[..., 2:3], 0.0, 1.0)
        self._num_of_vehicles.assign(tf.cast(tf.math.reduce_sum(mult_tensor), tf.int64))
        # self._vehicles.assign(vehicle_list * tf.expand_dims(mult_tensor, 1))
        # self._vehicles.assign(vehicle_list * mult_tensor)
        return vehicle_list * mult_tensor
        # staying_vehicles = tf.less(0.0, vehicle_list[..., 2])
        # staying_indices = tf.where(staying_vehicles)
        # new_vehicle_list = tf.gather(vehicle_list, staying_indices)
        # new_vehicle_list = tf.reshape(new_vehicle_list, [-1, 13])
        # num_of_vehicles = tf.shape(new_vehicle_list, out_type=tf.int64)[0]
        # self._num_of_vehicles.assign(num_of_vehicles)
        # padding = tf.convert_to_tensor([[0, self._capacity - tf.shape(new_vehicle_list, out_type=tf.int64)[0]],
        #                                 [0, 0]],
        #                                 tf.int64)
        # new_vehicle_list = tf.pad(new_vehicle_list, padding, mode='CONSTANT', constant_values=0.0, )
        # self._vehicles.assign(new_vehicle_list)


    # @tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32), tf.TensorSpec([], tf.int64)])
    # @tf.function(jit_compile=True)
    def _update_batch_vehicles(self, vehicles: tf.Tensor):
        #print('Tracing _update_batch_vehicles')
        priorities = tf.cast(vehicles[..., 5:7], tf.float32)
        charges = tf.cast(vehicles[..., 9:], tf.float32)
        priorities_sum = tf.reduce_sum(priorities, axis=0)
        charges_sum = tf.reduce_sum(charges, axis=0)
        num_of_cars = tf.cast(self._num_of_vehicles, tf.float32)
        mean_priorities = tf.math.divide_no_nan(priorities_sum, num_of_cars)
        # self._next_max_charge.assign(my_round(self._next_max_charge + charges_sum[0], tf.constant(2)))
        self._next_max_charge.assign(self._next_max_charge + charges_sum[0])
        # self._next_min_charge.assign(my_round(self._next_min_charge + charges_sum[1], tf.constant(2)))
        self._next_min_charge.assign(self._next_min_charge + charges_sum[1])
        # self._next_max_discharge.assign(my_round(self._next_max_discharge + charges_sum[2], tf.constant(2)))
        self._next_max_discharge.assign(self._next_max_discharge + charges_sum[2])
        # self._next_min_discharge.assign(my_round(self._next_min_discharge + charges_sum[3], tf.constant(2)))
        self._next_min_discharge.assign(self._next_min_discharge + charges_sum[3])
        # self._charge_mean_priority.assign(my_round(mean_priorities[0], tf.constant(3)))
        self._charge_mean_priority.assign(mean_priorities[0])
        # self._discharge_mean_priority.assign(my_round(mean_priorities[1], tf.constant(3)))
        self._discharge_mean_priority.assign(mean_priorities[1])

    # @tf.function
    def _update_parking_state(self, vehicles):
        #print('Tracing update_parking_state')
        # vehicles = self._vehicles
        num_of_vehicles = tf.cast(self._num_of_vehicles, tf.float32)
        priorities = vehicles[..., 5:7]
        current_charges = vehicles[..., 0]
        charges = vehicles[..., 9:]
        charges_sum = tf.reduce_sum(charges, axis=0)
        priorites_sum = tf.reduce_sum(priorities, axis=0)
        priorities_mean = tf.math.divide_no_nan(priorites_sum, num_of_vehicles)
        self._current_charge.assign(tf.reduce_sum(current_charges))
        # self._next_max_charge.assign(my_round(charges_sum[0], tf.constant(2)))
        self._next_max_charge.assign(charges_sum[0])
        # self._next_min_charge.assign(my_round(charges_sum[1], tf.constant(2)))
        self._next_min_charge.assign(charges_sum[1])
        # self._next_max_discharge.assign(my_round(charges_sum[2], tf.constant(2)))
        self._next_max_discharge.assign(charges_sum[2])
        # self._next_min_discharge.assign(my_round(charges_sum[3], tf.constant(2)))
        self._next_min_discharge.assign(charges_sum[3])
        # self._charge_mean_priority.assign(my_round(priorities_mean[0], tf.constant(3)))
        self._charge_mean_priority.assign(priorities_mean[0])
        # self._discharge_mean_priority.assign(my_round(priorities_mean[1], tf.constant(3)))
        self._discharge_mean_priority.assign(priorities_mean[1])

    # @tf.function
    def update_energy_state(self, charging_coefficient: tf.Tensor, vehicles: tf.Tensor):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        #print('Tracing update_energy_state')
        is_charging = tf.where(tf.less(0.0, charging_coefficient), 1.0, -1.0)
        vehicle_array = Vehicle.update_emergency_demand(vehicles)
        self._next_max_charge.assign_sub(self._next_min_charge)
        self._next_max_discharge.assign_sub(self._next_min_discharge)
        norm_constants = tf.math.divide_no_nan(tf.reduce_sum(vehicle_array[..., 9:12:2] * vehicle_array[..., 5:7],
                                                             axis=0),
                                               tf.stack((self._next_max_charge, self._next_max_discharge),
                                                        axis=0))
        normalization_constant = tf.where(tf.less(0.0, is_charging), norm_constants[0], norm_constants[1])
        # sorted_vehicles = self._sort_vehicles_for_charge_update(
        #     vehicle_array,
        #     charging_coefficient,
        #     normalization_constant)
        tensor_mapping = tf.where(tf.less(0.0, charging_coefficient), vehicle_array[..., 5], vehicle_array[..., 6])
        sorted_args = tf.argsort(tensor_mapping, direction='DESCENDING')
        sorted_vehicles = tf.gather(vehicle_array, sorted_args)
        new_vehicles = Vehicle.update_current_charge(charging_coefficient,
                                                     normalization_constant,
                                                     sorted_vehicles,
                                                     self._capacity_int32)
        # self._vehicles.assign(new_vehicles
        return new_vehicles

    def _sort_vehicles_by_departure(self, vehicle_array):
        #print('Tracing _sort_vehicles')
        sorted_args = tf.argsort(vehicle_array[..., 2], direction='DESCENDING')
        return tf.gather(vehicle_array, sorted_args)

    # def _sort_vehicles_by_departure(self, vehicle_array):
    #     #print('Tracing _sort_vehicles')
    #     departure_tensor = vehicle_array[..., 2]
    #     sorted_args = tf.argsort(departure_tensor, direction='DESCENDING')
    #     indice_tensor = tf.transpose(tf.stack((tf.range(self._capacity),
    #                                            tf.cast(sorted_args, tf.int64)),
    #                                           axis=0))
    #     permutation_sparse_matrix = tf.sparse.SparseTensor(indice_tensor,
    #                                                        tf.ones([self._capacity_int32], tf.float16),
    #                                                        [self._capacity_int32, self._capacity_int32])
    #     return_matrix = tf.matmul(tf.sparse.to_dense(permutation_sparse_matrix), vehicle_array, a_is_sparse=True)
    #     return return_matrix

    # def _sort_vehicles_for_charge_update(self,
    #                                      vehicle_array,
    #                                      charging_coefficient,
    #                                      normalization_constant):
    #     #print('Tracing _sort_vehicles_for_charge_update')
    #     tensor_mapping = tf.where(tf.less(zero_16, charging_coefficient), vehicle_array[..., 5], vehicle_array[..., 6])
    #     sorted_args = tf.argsort(tensor_mapping, direction='DESCENDING')
    #     return tf.gather(vehicle_array, sorted_args)

    # def _sort_vehicles_for_charge_update(self,
    #                                      vehicle_array,
    #                                      charging_coefficient,
    #                                      normalization_constant):
    #     #print('Tracing _sort_vehicles_for_charge_update')
    #     priorities = tf.cast(vehicle_array[..., 5:7], tf.float32)
    #     # tensor_1 = charging_coefficient * (1.0 + vehicle_array[..., 5] - normalization_constant)
    #     tensor_1 = charging_coefficient * (1.0 + priorities[..., 0] - normalization_constant)
    #     # tensor_2 = charging_coefficient * (1.0 + vehicle_array[..., 6] - normalization_constant)
    #     tensor_2 = charging_coefficient * (1.0 + priorities[..., 1] - normalization_constant)
    #     tensor_mapping = tf.where(tf.less(0.0, charging_coefficient), tensor_1, tensor_2)
    #     sorted_args = tf.argsort(tensor_mapping, direction='DESCENDING')
    #     indice_tensor = tf.transpose(tf.stack((tf.range(self._capacity),
    #                                            tf.cast(sorted_args, tf.int64)),
    #                                           axis=0))
    #     permutation_sparse_matrix = tf.sparse.SparseTensor(indice_tensor,
    #                                                        tf.ones([self._capacity], tf.float16),
    #                                                        [self._capacity, self._capacity])
    #     return_matrix = tf.matmul(tf.sparse.to_dense(permutation_sparse_matrix), vehicle_array, a_is_sparse=True)
    #     return return_matrix

    # @tf.function
    def update(self, charging_coefficient):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        #print('Tracing update (parking)')
        new_vehicles = self.update_energy_state(charging_coefficient, self._vehicles)
        staying_vehicles = self.depart_vehicles(new_vehicles)
        self._update_parking_state(staying_vehicles)
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
