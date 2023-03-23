import json
from typing import Any, Dict
import config as cfg
from app.models.tf_vehicle_5 import Vehicle
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

    _next_max_charge = tf.Variable(0.0, dtype=tf.float32)
    _next_min_charge = tf.Variable(0.0, dtype=tf.float32)
    _next_max_discharge = tf.Variable(0.0, dtype=tf.float32)
    _next_min_discharge = tf.Variable(0.0, dtype=tf.float32)
    _charge_mean_priority = tf.Variable(0.0, dtype=tf.float32)
    _discharge_mean_priority = tf.Variable(0.0, dtype=tf.float32)
    _current_charge = tf.Variable(0.0, dtype=tf.float32)
    _max_charging_rate = tf.constant(cfg.MAX_CHARGING_RATE, dtype=tf.float32)
    _max_discharging_rate = tf.constant(cfg.MAX_DISCHARGING_RATE, dtype=tf.float32)

    def __init__(self, capacity: int, name: str):
        self._capacity = tf.constant(capacity)
        self._vehicles = tf.Variable(tf.zeros(shape=(capacity, 13), dtype=tf.float32))
        self._num_of_vehicles = tf.Variable((0))
        self._vehicle_computer = Vehicle()
        self._capacity: tf.Tensor = tf.constant(capacity)
        self.name = name
        # self.

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
        print('Tracing assign_vehicle')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicles_added_number = tf.minimum(tf.shape(vehicles)[0], self._capacity - num_of_vehicles)
        vehicle_array = self._vehicles.value()
        new_vehicles = tf.gather(vehicles, tf.range(vehicles_added_number))
        new_charges = tf.reduce_sum(new_vehicles[..., 0])
        self._current_charge.assign_add(new_charges)
        parked_tensor = tf.map_fn(lambda t: self._vehicle_computer.park(self._max_charging_rate,
                                                                        self._max_discharging_rate,
                                                                        t),
                                  new_vehicles,
                                  parallel_iterations=20)
        # new_vehicle = self._vehicle_computer.park(self._max_charging_rate, self._max_discharging_rate, vehicle)
        # vehicle_array = tf.tensor_scatter_nd_update(vehicle_array,
        #                                             tf.reshape(num_of_vehicles, [1,1]),
        #                                             [new_vehicle])
        parked_tensor = tf.reshape(parked_tensor, [-1, 13])
        self._vehicles.scatter_nd_update(tf.reshape(tf.range(num_of_vehicles,
                                                             num_of_vehicles + vehicles_added_number), [-1,1]),
                                         # [*tf.unstack(parked_tensor)])
                                         parked_tensor)
        num_of_vehicles += vehicles_added_number
        self._num_of_vehicles.assign_add(vehicles_added_number)
        # sorted_array = self._sort_vehicles(vehicle_array[:num_of_vehicles])
        # self._vehicles.scatter_nd_update(tf.reshape(tf.range(num_of_vehicles), [-1, 1]), sorted_array)
        self._update_batch_vehicles(parked_tensor, num_of_vehicles)

    @tf.function
    def depart_vehicles(self):
        """
        Filter out all vehicles that have left the parking
        """
        print('Tracing depart_vehicles')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_list = self._vehicles.value()[:num_of_vehicles]
        staying_vehicles = tf.map_fn(lambda t: t[2] > 0.0, vehicle_list, fn_output_signature = tf.bool)
        staying_indices = tf.where(staying_vehicles)
        new_vehicle_list = tf.gather(vehicle_list, staying_indices)
        new_vehicle_list = tf.reshape(new_vehicle_list, [-1, 13])
        num_of_vehicles = tf.shape(new_vehicle_list)[0]
        self._num_of_vehicles.assign(num_of_vehicles)
        padding = tf.convert_to_tensor([[0, self._capacity - tf.shape(new_vehicle_list)[0]], [0, 0]], tf.int32)
        new_vehicle_list = tf.pad(new_vehicle_list, padding, mode='CONSTANT', constant_values=0.0)
        self._vehicles.assign(new_vehicle_list)

    # @tf.function
    # def _update_single_vehicle(self, vehicle: tf.Tensor, vehicles_counted: tf.Tensor):
    #     print('Tracing _update_single_vehicle')
    #     num_of_cars = vehicles_counted
    #     num_of_cars = tf.cast(num_of_cars, tf.float32)
    #
    #     next_max_charge = self._next_max_charge.value() + vehicle[9]
    #     next_min_charge = self._next_min_charge.value() + vehicle[10]
    #     next_max_discharge = self._next_max_discharge.value() + vehicle[11]
    #     next_min_discharge = self._next_min_discharge.value() + vehicle[12]
    #     next_max_charge = my_round(next_max_charge, tf.constant(2))
    #     next_min_charge = my_round(next_min_charge, tf.constant(2))
    #     next_max_discharge = my_round(next_max_discharge, tf.constant(2))
    #     next_min_discharge = my_round(next_min_discharge, tf.constant(2))
    #
    #     self._next_max_charge.assign(next_max_charge)
    #     self._next_min_charge.assign(next_min_charge)
    #     self._next_max_discharge.assign(next_max_discharge)
    #     self._next_min_discharge.assign(next_min_discharge)
    #
    #     charge_mean_priority = self._charge_mean_priority.value()
    #     discharge_mean_priority = self._discharge_mean_priority.value()
    #     charge_mean_priority = ((num_of_cars - 1.0) *
    #                                  charge_mean_priority +
    #                                  vehicle[5]) /\
    #                                  num_of_cars
    #     discharge_mean_priority = ((num_of_cars - 1.0) *
    #                                     discharge_mean_priority +
    #                                     vehicle[6]) /\
    #                                     num_of_cars
    #     charge_mean_priority = my_round(charge_mean_priority, tf.constant(3))
    #     discharge_mean_priority = my_round(discharge_mean_priority, tf.constant(3))
    #     self._charge_mean_priority.assign(charge_mean_priority)
    #     self._discharge_mean_priority.assign(discharge_mean_priority)

    @tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32), tf.TensorSpec([], tf.int32)])
    def _update_batch_vehicles(self, vehicles: tf.Tensor, vehicles_counted: tf.Tensor):
        print('Tracing _update_batch_vehicles')
        num_cars_added = tf.shape(vehicles)[0]
        num_of_cars = vehicles_counted - num_cars_added
        num_of_cars = tf.cast(num_of_cars, tf.float32)
        # tf.print(num_cars_added)
        c = lambda i, nmxc, nmnc, nmxd, nmnd, cmp, dmp: i < num_cars_added
        b = lambda i, nmxc, nmnc, nmxd, nmnd, cmp, dmp: (i + 1,
                                                         nmxc + vehicles[i][9],
                                                         nmnc + vehicles[i][10],
                                                         nmxd + vehicles[i][11],
                                                         nmnd + vehicles[i][12],
                                                         cmp + vehicles[i][5],
                                                         dmp + vehicles[i][6])
        starting_values = (tf.constant(0, tf.int32),
                           self._next_max_charge.value(),
                           self._next_min_charge.value(),
                           self._next_max_discharge.value(),
                           self._next_min_discharge.value(),
                           self._charge_mean_priority.value() * num_of_cars,
                           self._discharge_mean_priority.value() * num_of_cars)
        i, \
            next_max_charge, \
            next_min_charge, \
            next_max_discharge, \
            next_min_discharge, \
            charge_mean_priority, \
            discharge_mean_priority = tf.while_loop(c, b, starting_values, parallel_iterations=7)

        next_max_charge = my_round(next_max_charge, tf.constant(2))
        next_min_charge = my_round(next_min_charge, tf.constant(2))
        next_max_discharge = my_round(next_max_discharge, tf.constant(2))
        next_min_discharge = my_round(next_min_discharge, tf.constant(2))
        charge_mean_priority = my_round(charge_mean_priority / tf.cast(vehicles_counted, tf.float32),
                                        tf.constant(3))
        discharge_mean_priority = my_round(discharge_mean_priority / tf.cast(vehicles_counted, tf.float32),
                                           tf.constant(3))
        self._next_max_charge.assign(next_max_charge)
        self._next_min_charge.assign(next_min_charge)
        self._next_max_discharge.assign(next_max_discharge)
        self._next_min_discharge.assign(next_min_discharge)
        self._charge_mean_priority.assign(charge_mean_priority)
        self._discharge_mean_priority.assign(discharge_mean_priority)


    @tf.function
    def _update_parking_state(self):
        print('Tracing update_parking_state')
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_array = self._vehicles.value()[:num_of_vehicles]
        c = lambda i, max_c, min_c, max_d, min_d, cmp, dmp: i < num_of_vehicles
        b = lambda i, max_c, min_c, max_d, min_d, cmp, dmp: (
            i + 1,
            max_c + vehicle_array[i][9],
            min_c + vehicle_array[i][10],
            max_d + vehicle_array[i][11],
            min_d + vehicle_array[i][12],
            cmp + vehicle_array[i][5],
            dmp + vehicle_array[i][6],
        )
        i, \
            next_max_charge, \
            next_min_charge, \
            next_max_discharge, \
            next_min_discharge, \
            charge_mean_priority, \
            discharge_mean_priority = tf.while_loop(c, b, (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), parallel_iterations=7)
        next_max_charge = my_round(next_max_charge, tf.constant(2))
        next_min_charge = my_round(next_min_charge, tf.constant(2))
        next_max_discharge = my_round(next_max_discharge, tf.constant(2))
        next_min_discharge = my_round(next_min_discharge, tf.constant(2))
        charge_mean_priority = my_round(charge_mean_priority / tf.cast(num_of_vehicles, tf.float32), tf.constant(3))
        discharge_mean_priority = my_round(discharge_mean_priority / tf.cast(num_of_vehicles, tf.float32), tf.constant(3))
        self._next_max_charge.assign(next_max_charge)
        self._next_min_charge.assign(next_min_charge)
        self._next_max_discharge.assign(next_max_discharge)
        self._next_min_discharge.assign(next_min_discharge)
        self._charge_mean_priority.assign(charge_mean_priority)
        self._discharge_mean_priority.assign(discharge_mean_priority)

    @tf.function
    def _calculate_normalization_constant(self):
        print('Tracing _calculate_normalization_constant')
        normalization_charge_constant = tf.constant(0.0)
        normalization_discharge_constant = tf.constant(0.0)
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_array = self._vehicles.value()[:num_of_vehicles]
        c = lambda i, ncc, ndc: i < num_of_vehicles
        b = lambda i, ncc, ndc: (i + 1,
                                 ncc + vehicle_array[i][9] * vehicle_array[i][5],
                                 ndc + vehicle_array[i][11] * vehicle_array[i][6])
        _,\
            normalization_charge_constant, \
            normalization_discharge_constant = tf.while_loop(c, b, (0, 0.0, 0.0), parallel_iterations=3)

        if self._next_max_charge.value() != 0.0:
            normalization_charge_constant = my_round(normalization_charge_constant /
                                                     self._next_max_charge.value(),
                                                     tf.constant(3))

        if self._next_max_discharge.value() != 0.0:
            normalization_discharge_constant = my_round(normalization_discharge_constant /
                                                        self._next_max_discharge.value(),
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
        print('Tracing update_energy_state')
        is_charging = tf.cond(tf.less(0.0, charging_coefficient),
                              lambda: True,
                              lambda: False)
        sign = tf.cond(is_charging,
                       lambda: tf.constant(1.0, dtype=tf.float32),
                       lambda: tf.constant(-1.0, dtype=tf.float32))

        new_list = tf.constant([])
        num_of_vehicles = self._num_of_vehicles.value()
        vehicle_array = self._vehicles.value()[:num_of_vehicles]
        for vehicle in vehicle_array:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(new_list, tf.TensorShape([None]))]
            )
            new_vehicle = self._vehicle_computer.update_emergency_demand(vehicle)
            new_list = tf.concat((new_list, new_vehicle), axis=0)
        vehicle_array = tf.reshape(new_list, shape=[-1,13])
        self._next_max_charge.assign_sub(self._next_min_charge.value())
        self._next_max_discharge.assign_sub(self._next_min_discharge.value())

        normalization_charge_constant, normalization_discharge_constant = self._calculate_normalization_constant()
        normalization_constant = normalization_charge_constant if is_charging else normalization_discharge_constant

        sorted_vehicles = self._sort_vehicles_for_charge_update(
            vehicle_array,
            sign,
            charging_coefficient,
            normalization_constant,
            is_charging
        )
        new_list = tf.constant([])
        residue = tf.constant(0.0)
        for vehicle in sorted_vehicles:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(new_list, tf.TensorShape([None]))]
            )
            residue, new_vehicle = self._vehicle_computer.update_current_charge(
                charging_coefficient, normalization_constant, residue, vehicle
            )
            new_list = tf.concat((new_list, new_vehicle), axis=0)
        vehicle_array = tf.reshape(new_list, shape=[-1, 13])
        # vehicle_array = self._sort_vehicles(new_list)
        num_of_vehicles = tf.shape(vehicle_array)[0]
        self._vehicles.scatter_nd_update(tf.reshape(tf.range(num_of_vehicles), [-1, 1]), vehicle_array)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32)])
    def _sort_vehicles(self, vehicle_array):
        print('Tracing _sort_vehicles')
        departure_tensor = vehicle_array[..., 2]
        sorted_args = tf.argsort(departure_tensor)
        return tf.gather(vehicle_array, sorted_args)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 13], dtype=tf.float32, name='vehicle_array'),
                                  tf.TensorSpec(shape=(), dtype=tf.float32, name='sign'),
                                  tf.TensorSpec(shape=(), dtype=tf.float32, name='charging_coefficient'),
                                  tf.TensorSpec(shape=(), dtype=tf.float32, name='normalization_constant'),
                                  tf.TensorSpec(shape=(), dtype=tf.bool, name='is_charging'),])
    def _sort_vehicles_for_charge_update(self,
                                         vehicle_array,
                                         sign,
                                         charging_coefficient,
                                         normalization_constant,
                                         is_charging):
        print('Tracing _sort_vehicles_for_charge_update')
        if is_charging:
            tensor_mapping = tf.map_fn(
                lambda t: sign * charging_coefficient * (1.0 + t[5] - normalization_constant),
                vehicle_array
            )
        else:
            tensor_mapping = tf.map_fn(
                lambda t: sign * charging_coefficient * (1.0 + t[6] - normalization_constant),
                vehicle_array
            )
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
        print('Tracing update (parking)')
        self.update_energy_state(charging_coefficient)
        self.depart_vehicles()
        self._update_parking_state()

    def toJson(self) -> Dict[str, Any]:
        return {
            "class": Parking.__name__,
            "name": self.name,
            "max_charging_rage": self._max_charging_rate,
            "max_discharging_rate": self._max_discharging_rate,
            "vehicles": list(map(lambda v: v.toJson(), self._vehicles)),
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
