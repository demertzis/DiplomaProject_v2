import json
from typing import Any, Dict, List, Optional
import config as cfg
from app.error_handling import ParkingIsFull
from app.models.tf_vehicle_4 import Vehicle, VehicleFields
from app.models.tf_utils import my_round

from functools import reduce

import tensorflow as tf


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

    _next_max_charge: float = 0.0
    _next_min_charge: float = 0.0
    _next_max_discharge: float = 0.0
    _next_min_discharge: float = 0.0
    _charge_mean_priority: float = 0.0
    _discharge_mean_priority: float = 0.0
    _current_charge: float = 0.0
    _max_charging_rate: int = cfg.MAX_CHARGING_RATE
    _max_discharging_rate: int = cfg.MAX_DISCHARGING_RATE

    def __init__(self, capacity: int, name: str):
        _vehicles: List[VehicleFields] = []
        self._vehicles = _vehicles
        self._vehicle_computer = Vehicle()
        self._capacity = capacity
        self.name = name

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

    def assign_vehicle(self, vehicle: VehicleFields):
        """
        Assign vehicle to a parking space

        ### Arguments:
            vehicle (``Vehicle``) :
                description: A non-parked instance of the vehicle class

        ### Raises:
            ParkingIsFull: The parking has no free spaces
        """
        num_of_vehicles = len(self._vehicles)
        if num_of_vehicles == self._capacity:
            raise ParkingIsFull()

        self._current_charge += vehicle.current_charge
        # vehicle.park(self._max_charging_rate, self._max_discharging_rate)
        self._vehicle_computer.assign_fields(vehicle)
        self._vehicle_computer.park(self._max_charging_rate, self._max_discharging_rate)
        new_vehicle = self._vehicle_computer.return_fields()
        self._vehicles.append(new_vehicle)
        self._vehicles.sort(key=lambda v: v.time_before_departure)
        self._update_single_vehicle(new_vehicle)

    def depart_vehicles(self):
        """
        Filter out all vehicles that have left the parking
        """
        # departed_vehicles = list(filter(lambda v: v.get_time_before_departure() == 0, self._vehicles))
        self._vehicles = list(filter(lambda v: v.time_before_departure > 0, self._vehicles))

        # overcharged_time = []
        # for v in departed_vehicles:
        #     # print("Vehicle with target ", v.get_target_charge(), " and changes ", v._changes)
        #     overcharged_time.append(v.get_overchared_time())

        # return overcharged_time
    def _update_single_vehicle(self, vehicle: Vehicle, vehicles_counted: Optional[int] = None):
        num_of_cars = vehicles_counted or len(self._vehicles)
        self._next_max_charge += vehicle.next_max_charge
        self._next_min_charge += vehicle.next_min_charge
        self._next_max_discharge += vehicle.next_max_discharge
        self._next_min_discharge += vehicle.next_min_discharge

        self._next_max_charge = my_round(self._next_max_charge, tf.constant(2))
        self._next_min_charge = my_round(self._next_min_charge, tf.constant(2))
        self._next_max_discharge = my_round(self._next_max_discharge, tf.constant(2))
        self._next_min_discharge = my_round(self._next_min_discharge, tf.constant(2))


        self._charge_mean_priority = ((num_of_cars - 1) *
                                     self._charge_mean_priority +
                                     vehicle.charge_priority) /\
                                     num_of_cars
        self._discharge_mean_priority = ((num_of_cars - 1) *
                                        self._charge_mean_priority +
                                        vehicle.discharge_priority) /\
                                        num_of_cars
        self._charge_mean_priority = my_round(self._charge_mean_priority, tf.constant(3))
        self._discharge_mean_priority = my_round(self._discharge_mean_priority, tf.constant(3))




    def _update_parking_state(self):
        self._next_max_charge = 0.0
        self._next_min_charge = 0.0
        self._next_max_discharge = 0.0
        self._next_min_discharge = 0.0
        self._charge_mean_priority = 0.0
        self._discharge_mean_priority = 0.0
        i = 1
        for vehicle in self._vehicles:
            self._update_single_vehicle(vehicle, i)
            i += 1

    def _calculate_normalization_constant(self):
        normalization_charge_constant = 0.0
        normalization_discharge_constant = 0.0
        for vehicle in self._vehicles:
            next_max_charge = vehicle.next_max_charge
            next_max_discharge = vehicle.next_max_discharge
            charge_priority = vehicle.charge_priority
            discharge_priority = vehicle.discharge_priority
            normalization_charge_constant += next_max_charge * charge_priority
            normalization_discharge_constant += next_max_discharge * discharge_priority

        if self._next_max_charge != 0:
            normalization_charge_constant = my_round(normalization_charge_constant / self._next_max_charge,
                                                     tf.constant(3))

        if self._next_max_discharge != 0:
            normalization_discharge_constant = my_round(normalization_discharge_constant / self._next_max_discharge,
                                                        tf.constant(3))

        return normalization_charge_constant, normalization_discharge_constant

    def update_energy_state(self, charging_coefficient: float):
        """
        Update energy state of parking

        ### Arguments:
            charging_coefficient (``float``) :
                description: The ratio of the used charging/discharging capacity
        """
        is_charging = charging_coefficient > 0
        sign = tf.cond(is_charging,
                       lambda: tf.constant(1.0, dtype=tf.float32),
                       lambda: tf.constant(-1.0, dtype=tf.float32))

        new_list = []
        for vehicle in self._vehicles:
            self._vehicle_computer.assign_fields(vehicle)
            self._vehicle_computer.update_emergency_demand()
            new_list.append(self._vehicle_computer.return_fields())
        self._vehicles = new_list
            # Vehicle.update_emergency_demand(vehicle)

        self._next_max_charge -= self._next_min_charge
        self._next_max_discharge -= self._next_min_discharge

        normalization_charge_constant, normalization_discharge_constant = self._calculate_normalization_constant()
        normalization_constant = (
            normalization_charge_constant if is_charging else normalization_discharge_constant
        )

        # residue = 0.0
        # for vehicle in sorted(
        #     self._vehicles,
        #     key=lambda v: sign * charging_coefficient * (1.0 + tf.cond(is_charging,
        #                                                              lambda: v.get_charge_priority(),
        #                                                              lambda: v.get_discharge_priority())
        #                                                  - normalization_constant),
        #     reverse=True,
        # ):
        #     residue = vehicle.update_current_charge(
        #         charging_coefficient, normalization_constant, residue
        #     )

        # sort_key_tensor = tf.map_fn(fn = lambda v: sign * charging_coefficient * (1.0 + tf.cond(is_charging,
        #                                                              lambda: v.get_charge_priority(),
        #                                                              lambda: v.get_discharge_priority())
        #                                                  - normalization_constant),
        #                             elems = self._vehicles,)
        # sort_key = lambda v: sign * charging_coefficient * (1.0 + tf.cond(is_charging,
        #                                                              lambda: v.get_charge_priority(),
        #                                                              lambda: v.get_discharge_priority())
        #                                                  - normalization_constant)
        # sort_key_tensor = tf.convert_to_tensor(tuple(map(sort_key, self._vehicles)),
        #                                        dtype=tf.float32)
        # sorted_vehicle_indices = tf.argsort(values=sort_key_tensor,
        #                                     direction='DESCENDING',)
        # residue = tf.constant(0.0, dtype=tf.float32)
        # sorted_vehicle_list =

        # sorted_vehicles = sorted(
        #     self._vehicles,
        #     key=lambda v: sign * charging_coefficient * (1.0 + tf.cond(is_charging,
        #                                                                lambda: v.get_charge_priority(),
        #                                                                lambda: v.get_discharge_priority())
        #                                                  - normalization_constant),
        #     reverse=True,
        # )
        if is_charging:
            sorted_vehicles = sorted(
                self._vehicles,
                key=lambda v: sign * charging_coefficient * (1.0 + v.charge_priority - normalization_constant),
                reverse=True,
            )
        else:
            sorted_vehicles = sorted(
                self._vehicles,
                key=lambda v: sign * charging_coefficient * (1.0 + v.discharge_priority - normalization_constant),
                reverse=True,
            )

        # reduce(lambda a, v: v.update_current_charge(charging_coefficient,
        #                                             normalization_constant,
        #                                             a),
        # reduce(lambda a, v: Vehicle.update_current_charge(v, charging_coefficient,
        #                                             normalization_constant,
        #                                             a),
        #        sorted_vehicles,
        #        tf.constant(0.0, dtype=tf.float32))
        new_list = []
        residue = 0.0
        for vehicle in sorted_vehicles:
            self._vehicle_computer.assign_fields(vehicle)
            residue = self._vehicle_computer.update_current_charge(
                charging_coefficient, normalization_constant, residue
            )
            new_list.append(self._vehicle_computer.return_fields())
        self._vehicles = new_list
        self._vehicles.sort(key=lambda v: v.time_before_departure)



    def update(self, charging_coefficient):
        """
        Given the action input, it performs an update step

        ### Arguments:
            charging_coefficient (``float``) :
                description: The charging coefficient
        """
        self.update_energy_state(charging_coefficient)
        self.depart_vehicles()
        self._update_parking_state()
        # return avg_charge_levels, overcharged_time

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
