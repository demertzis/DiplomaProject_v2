import json
from typing import Any, Dict
import config as cfg
import app.models.tf_vehicle_6 as Vehicle
from app.models.tf_utils import my_round


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
    _max_charging_rate = tf.constant(cfg.MAX_CHARGING_RATE, dtype=tf.float32)
    _max_discharging_rate = tf.constant(cfg.MAX_DISCHARGING_RATE, dtype=tf.float32)

    def __init__(self, capacity: int, name: str):
        self._capacity = tf.constant(capacity)
        self._vehicles = tf.Variable(tf.zeros(shape=(capacity, 13), dtype=tf.float32), trainable=False)
        self._num_of_vehicles = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._capacity: tf.Tensor = tf.constant(capacity, dtype=tf.int64)
        self.name = name

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
                             self._max_charging_rate,
                             self._max_discharging_rate)

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

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32)])
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
        vehicles_added_number = tf.minimum(tf.shape(vehicles, tf.int64)[0], self._capacity - num_of_vehicles)
        new_vehicles = tf.gather(vehicles, tf.range(vehicles_added_number))
        new_charges = tf.reduce_sum(new_vehicles[..., 0])
        self._current_charge.assign_add(new_charges)
        parked_tensor = Vehicle.park(self._max_charging_rate,
                                     self._max_discharging_rate,
                                     new_vehicles)
        self._vehicles.scatter_nd_update(tf.expand_dims(tf.range(num_of_vehicles,
                                                                 num_of_vehicles + vehicles_added_number),
                                                        axis=1),
                                         parked_tensor)
        num_of_vehicles += vehicles_added_number
        self._num_of_vehicles.assign_add(vehicles_added_number)
        self._update_batch_vehicles(parked_tensor, num_of_vehicles)

    @tf.function
    def depart_vehicles(self):
        """
        Filter out all vehicles that have left the parking
        """
        #print('Tracing depart_vehicles')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_list = self._vehicles.value()[:num_of_vehicles]
        staying_vehicles = tf.less(0.0, vehicle_list[..., 2])
        staying_indices = tf.where(staying_vehicles)
        new_vehicle_list = tf.gather(vehicle_list, staying_indices)
        new_vehicle_list = tf.reshape(new_vehicle_list, [-1, 13])
        num_of_vehicles = tf.shape(new_vehicle_list, out_type=tf.int64)[0]
        self._num_of_vehicles.assign(num_of_vehicles)
        padding = tf.convert_to_tensor([[0, self._capacity - tf.shape(new_vehicle_list, out_type=tf.int64)[0]],
                                        [0, 0]],
                                        tf.int64)
        new_vehicle_list = tf.pad(new_vehicle_list, padding, mode='CONSTANT', constant_values=0.0, )
        self._vehicles.assign(new_vehicle_list)


    @tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32), tf.TensorSpec([], tf.int64)])
    def _update_batch_vehicles(self, vehicles: tf.Tensor, vehicles_counted: tf.Tensor):
        #print('Tracing _update_batch_vehicles')
        num_cars_added = tf.shape(vehicles, out_type=tf.int64)[0]
        num_of_cars = vehicles_counted - num_cars_added
        num_of_cars = tf.cast(num_of_cars, tf.float32)
        priorities = vehicles[..., 5:7]
        charges = vehicles[..., 9:]
        priorites_sum = tf.reduce_sum(priorities, axis=0)
        charges_sum = tf.reduce_sum(charges, axis=0)
        charge_priorities_sum = tf.math.divide_no_nan(priorites_sum[0] + self._charge_mean_priority * num_of_cars,
                                                      tf.cast(vehicles_counted, tf.float32))
        discharge_priorities_sum = tf.math.divide_no_nan(priorites_sum[1] + self._charge_mean_priority * num_of_cars,
                                                         tf.cast(vehicles_counted, tf.float32))
        self._next_max_charge.assign(my_round(self._next_max_charge + charges_sum[0], tf.constant(2)))
        self._next_min_charge.assign(my_round(self._next_min_charge + charges_sum[1], tf.constant(2)))
        self._next_max_discharge.assign(my_round(self._next_max_discharge + charges_sum[2], tf.constant(2)))
        self._next_min_discharge.assign(my_round(self._next_min_discharge + charges_sum[3], tf.constant(2)))
        self._charge_mean_priority.assign(my_round(charge_priorities_sum, tf.constant(3)))
        self._discharge_mean_priority.assign(my_round(discharge_priorities_sum, tf.constant(3)))

    @tf.function
    def _update_parking_state(self):
        #print('Tracing update_parking_state')
        num_of_vehicles = self._num_of_vehicles
        vehicles = self._vehicles.gather_nd(tf.expand_dims(tf.range(num_of_vehicles), axis=1))
        num_of_vehicles = tf.cast(num_of_vehicles, tf.float32)
        priorities = vehicles[..., 5:7]
        charges = vehicles[..., 9:]
        priorites_sum = tf.reduce_sum(priorities, axis=0)
        charges_sum = tf.reduce_sum(charges, axis=0)
        charge_priorities_sum = tf.math.divide_no_nan(priorites_sum[0], num_of_vehicles)
        discharge_priorities_sum = tf.math.divide_no_nan(priorites_sum[1], num_of_vehicles)
        self._next_max_charge.assign(my_round(charges_sum[0], tf.constant(2)))
        self._next_min_charge.assign(my_round(charges_sum[1], tf.constant(2)))
        self._next_max_discharge.assign(my_round(charges_sum[2], tf.constant(2)))
        self._next_min_discharge.assign(my_round(charges_sum[3], tf.constant(2)))
        self._charge_mean_priority.assign(my_round(charge_priorities_sum, tf.constant(3)))
        self._discharge_mean_priority.assign(my_round(discharge_priorities_sum, tf.constant(3)))

    @tf.function
    def _calculate_normalization_constant(self):
        #print('Tracing _calculate_normalization_constant')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_array = self._vehicles.value()[:num_of_vehicles]
        needed_fields_1 = tf.gather(vehicle_array, [9, 11], axis=1)
        needed_fields_2 = tf.gather(vehicle_array, [5, 6], axis=1)
        norm_constants = tf.reduce_sum(needed_fields_1 * needed_fields_2, axis=0)
        normalization_charge_constant = norm_constants[0]
        normalization_discharge_constant = norm_constants[1]
        normalization_charge_constant = my_round(tf.math.divide_no_nan(normalization_charge_constant,
                                                                       self._next_max_charge),
                                                 tf.constant(3))
        normalization_discharge_constant = my_round(tf.math.divide_no_nan(normalization_discharge_constant,
                                                                          self._next_max_discharge),
                                                    tf.constant(3))

        return normalization_charge_constant, normalization_discharge_constant

    @tf.function
    def update_energy_state(self, charging_coefficient: tf.Tensor):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        #print('Tracing update_energy_state')
        is_charging = tf.where(tf.less(0.0, charging_coefficient), 1.0, -1.0)
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_array = self._vehicles.value()[:num_of_vehicles]
        vehicle_array = Vehicle.update_emergency_demand(vehicle_array)
        self._next_max_charge.assign_sub(self._next_min_charge.value())
        self._next_max_discharge.assign_sub(self._next_min_discharge.value())

        normalization_charge_constant, normalization_discharge_constant = self._calculate_normalization_constant()
        normalization_constant = tf.math.maximum(normalization_charge_constant * is_charging,
                                                 (-1.0) * normalization_discharge_constant * is_charging)
        sorted_vehicles = self._sort_vehicles_for_charge_update(
            vehicle_array,
            charging_coefficient,
            normalization_constant,
        )
        new_vehicles = Vehicle.update_current_charge(charging_coefficient, normalization_constant, sorted_vehicles)
        num_of_vehicles = tf.shape(vehicle_array)[0]
        self._vehicles.scatter_nd_update(tf.reshape(tf.range(num_of_vehicles), [-1, 1]), new_vehicles)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32)])
    def _sort_vehicles(self, vehicle_array):
        #print('Tracing _sort_vehicles')
        departure_tensor = vehicle_array[..., 2]
        sorted_args = tf.argsort(departure_tensor)
        return tf.gather(vehicle_array, sorted_args)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32, name='vehicle_array'),
                                  tf.TensorSpec(shape=(), dtype=tf.float32, name='charging_coefficient'),
                                  tf.TensorSpec(shape=(), dtype=tf.float32, name='normalization_constant')])
    def _sort_vehicles_for_charge_update(self,
                                         vehicle_array,
                                         charging_coefficient,
                                         normalization_constant):
        #print('Tracing _sort_vehicles_for_charge_update')
        tensor_1 = charging_coefficient * (1.0 + vehicle_array[..., 5] - normalization_constant)
        tensor_2 = charging_coefficient * (1.0 + vehicle_array[..., 6] - normalization_constant)
        tensor_mapping = tf.where(tf.less(0.0, charging_coefficient), tensor_1, tensor_2)
        sorted_args = tf.argsort(tensor_mapping, direction='DESCENDING')
        return tf.gather(vehicle_array, sorted_args)

    @tf.function
    def update(self, charging_coefficient):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        #print('Tracing update (parking)')
        self.update_energy_state(charging_coefficient)
        self.depart_vehicles()
        self._update_parking_state()

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
