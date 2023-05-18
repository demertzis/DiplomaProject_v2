from tf_agents.typing import types
from tf_agents.utils import common

from app.models.tf_utils import tf_find_intersection_v2, my_pad
import json
from typing import Any, Dict, NamedTuple
import config
import tensorflow as tf
import tensorflow_probability as tfp

from app.models.tf_utils import my_round


class VehicleFields(tf.experimental.ExtensionType):
    """
    Used as a carrier of Vehicle State in order to use the Vehicle class methods in graph mode without
    retracing for every car (which  makes it slower than eager execution). The way it works is that we
    save VehicleField objects (which are ExtensionType) as the vehicles in the parking while the previous
    Vehicle class now saves as a computer for vehicle operations by loading to it's tf.Variables the VehicleField
    properties (Vehicle.assign). After the operation has been
    """
    _current_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _target_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _time_before_departure: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _max_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _min_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _charge_priority: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _discharge_priority: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _max_charging_rate: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _max_discharging_rate: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _next_max_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _next_min_charge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _next_max_discharge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
    _next_min_discharge: tf.Tensor = tf.constant(0.0, dtype=tf.float32)

    @property
    def current_charge(self):
        return self._current_charge
    @property
    def target_charge(self):
        return self._target_charge
    @property
    def time_before_departure(self):
        return self._time_before_departure
    @property
    def max_charge(self):
        return self._max_charge
    @property
    def min_charge(self):
        return self._min_charge
    @property
    def charge_priority(self):
        return self._charge_priority
    @property
    def discharge_priority(self):
        return self._discharge_priority
    @property
    def max_charging_rate(self):
        return self._max_charging_rate
    @property
    def max_discharging_rate(self):
        return self._max_discharging_rate
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

    def return_as_tensor(self):
        return tf.stack((
            self._current_charge,
            self._target_charge,
            self._time_before_departure,
            self._max_charge,
            self._min_charge,
            self._charge_priority,
            self._discharge_priority,
            self._max_charging_rate,
            self._max_discharging_rate,
            self._next_max_charge,
            self._next_min_charge,
            self._next_max_discharge,
            self._next_min_discharge,),
            axis=0,)


class Vehicle:
    """
    A class to represent a vehicle and its charging functionalities

    ### Arguments:
        initial_charge (``float``) :
            description: The initial charge of the vehicle's battery
        target_charge (``float``) :
            description: The desired charge to be achieved in `total_stay` hours
        total_stay (``int``) :
            description: The total hours that the vehicle will remain in the parking
        max_charge (``float``) :
            description: The maximum allowed charge to be stored in the vehicle's battery
        min_charge (``float``) :
            description: The minimum allowed charge to be stored in the vehicle's battery

    ### Attributes:
        _charge_priority (``float``) :
            description: The charge priority of the vehicle
        _discharge_priority (``float``) :
            description: The discharge priority of the vehicle
        _next_max_charge (``float``) :
            description: The maximum charging state needed to be achieved in the next hour so tha
            the target_charge is achievable
        _next_min_charge (``float``) :
            description: The minimum charging state needed to be achieved in the next hour so that
            the target_charge is achievable
    """
    def __init__(self):
        self._current_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._target_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._time_before_departure: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._max_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._min_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._charge_priority: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._discharge_priority: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._max_charging_rate: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._max_discharging_rate: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._next_max_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._next_min_charge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._next_max_discharge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)
        self._next_min_discharge: tf.Tensor = tf.Variable(0.0, dtype=tf.float32)

    def assign_fields(self, fields):
        self._current_charge.assign(fields.current_charge)
        self._target_charge.assign(fields.target_charge)
        self._time_before_departure.assign(tf.cast(fields.time_before_departure, tf.float32))
        self._max_charge.assign(fields.max_charge)
        self._min_charge.assign(fields.min_charge)
        self._charge_priority.assign(fields.charge_priority)
        self._discharge_priority.assign(fields.discharge_priority)
        self._max_charging_rate.assign(fields.max_charging_rate)
        self._max_discharging_rate.assign(fields.max_discharging_rate)
        self._next_max_charge.assign(fields.next_max_charge)
        self._next_min_charge.assign(fields.next_min_charge)
        self._next_max_discharge.assign(fields.next_max_discharge)
        self._next_min_discharge.assign(fields.next_min_discharge)

    @tf.function
    def return_fields(self):
        return VehicleFields(self._current_charge.value(),
                             self._target_charge.value(),
                             self._time_before_departure.value(),
                             self._max_charge.value(),
                             self._min_charge.value(),
                             self._charge_priority.value(),
                             self._discharge_priority.value(),
                             self._max_charging_rate.value(),
                             self._max_discharging_rate.value(),
                             self._next_max_charge.value(),
                             self._next_min_charge.value(),
                             self._next_max_discharge.value(),
                             self._next_min_discharge.value())

    @tf.function
    def park(self, max_charging_rate: tf.Tensor, max_discharging_rate: tf.Tensor):
        """
        Park vehicle to parking space and initialize its state variables
        To be called by the parking space it was assigned to

        ### Arguments:
            max_charging_rate (``float``):
                description: The maximum charging rate
            max_discharging_rate (``float``):
                description: The maximum discharging rate
        """
        print('Tracing park')
        self._max_charging_rate.assign(max_charging_rate)
        self._max_discharging_rate.assign(max_discharging_rate)
        self.update()

    @tf.function
    def _calculate_next_max_charge(self, current_charge: tf.Tensor, time_before_departure: tf.Tensor):
        """
        Calculate next max charge based on the below formula

        min (
            ``max_charging_rate (kWh) * 1 hour + current_charge,``\n
            ``max_charge,``\n
            ``(time_of_departure - time_has_past - 1) * max_discharging_rate + target_charge,``\n
        )

        Note: ``(time_of_departure - time_has_past - 1) = time_before_departure - 1``

        ### Arguments:
            current_charge (``float``):
                description: The current charge of the vehicle's battery
            time_before_departure (``int``):
                description: The total hours remaining before departurep

        ### Returns:
            float : The next max charge
        """
        print('Tracing calculate_next_max_charge')
        return my_round(
            tf.math.reduce_min(
                tf.stack((self._max_charging_rate + current_charge,
                          self._max_charge,
                          (time_before_departure - 1.0) *\
                              self._max_discharging_rate +\
                              self._target_charge,),
                          axis=0)
                ),
            tf.constant(3),
            )

    @tf.function
    def _calculate_next_min_charge(self, current_charge: tf.Tensor, time_before_departure: tf.Tensor):
        """
        Calculate next min charge based on the below formula

        max (
            ``current_charge - max_discharging_rate (kWh) * 1 hour,``\n
            ``min_charge,``\n
            ``target_charge - (time_of_departure - time_has_past - 1) * max_charging_rate,``\n
        )

        Note: ``(time_of_departure - time_has_past - 1) = time_before_departure - 1``

        ### Arguments:
            current_charge (``float``):
                description: The current charge of the vehicle's battery
            time_before_departure (``int``):
                description: The total hours remaining before departure

        ### Returns:
            float : The next min charge
        """
        print('Tracing calculate_next_min_charge')
        return my_round(
            tf.math.reduce_max(
                tf.stack(
                    (current_charge - self._max_discharging_rate,
                     self._min_charge,
                     self._target_charge - \
                         (time_before_departure - 1) *\
                         self._max_charging_rate,),
                    axis=0),
            ),
            tf.constant(3),
        )

    @tf.function
    def _update_next_charging_states(self):
        """
        Update max and min charge state variables
        - ``ΔΕ+(max) = max(0, next_max_charge - current_charge)``
        - ``ΔΕ+(min) = max(0, next_min_charge - current_charge)``
        - ``ΔΕ-(max) = max(0, current_charge - next_min_charge)``
        - ``ΔΕ-(min) = max(0, current_charge - next_max_charge)``
        """
        print('Tracing update_next_charging_states')
        next_max_charge = self._calculate_next_max_charge(self._current_charge.value(),
                                                          self._time_before_departure.value())
        next_min_charge = self._calculate_next_min_charge(self._current_charge.value(),
                                                          self._time_before_departure.value())

        new_max_charge = my_round(tf.math.maximum(0.0, next_max_charge - self._current_charge), tf.constant(3))
        self._next_max_charge.assign(new_max_charge)
        new_min_charge = my_round(tf.math.maximum(0.0, next_min_charge - self._current_charge), tf.constant(3))
        self._next_min_charge.assign(new_min_charge)
        new_max_discharge = my_round(tf.math.maximum(0.0, self._current_charge - next_min_charge), tf.constant(3))
        self._next_max_discharge.assign(new_max_discharge)
        new_min_discharge = my_round(tf.math.maximum(0.0, self._current_charge - next_max_charge), tf.constant(3))
        self._next_min_discharge.assign(new_min_discharge)
    @tf.function
    def _calculate_charge_curves(self):
        """
        Calculate the max and min charge curves of the vehicle

        The max curve is the curve describing the maximum possible charge the vehicle can achieve at each timestamp
        so that the target charge is achievable given a initial charge, a max charging rate, a max discharging rate,
        a max charge and the time before departure

        The min curve is the curve describing the minimum possible charge the vehicle can achieve at each timestamp
        so that the target charge is achievable given a initial charge, a max charging rate, a max discharging rate,
        a min charge and the time before departure

        ### Returns
            Tuple[float[], float[]] : The points of the max and min curve respectively in ascending time order
        """
        print('Tracing calculate_charge_curves')
        current_max_charge = self._current_charge.value()
        current_min_charge = self._current_charge.value()

        max_charges = tf.TensorArray(dtype=tf.float32,
                                     size=tf.cast(self._time_before_departure, tf.int32) +1,
                                     clear_after_read=False).write(0, current_max_charge)

        min_charges = tf.TensorArray(dtype=tf.float32,
                                     size=tf.cast(self._time_before_departure, tf.int32) + 1,
                                     clear_after_read=False).write(0, current_min_charge)
        i = tf.constant(1)
        for t in tf.range(self._time_before_departure, 0.0, -1.0):
            current_max_charge = self._calculate_next_max_charge(current_max_charge, t)
            current_min_charge = self._calculate_next_min_charge(current_min_charge, t)

            max_charges = max_charges.write(i, current_max_charge)
            min_charges = min_charges.write(i, current_min_charge)
            i += 1
        return max_charges.stack(), min_charges.stack()

    @tf.function
    def _update_priorities(self):
        """
        Update the charging and discharging priorities of the vehicle

        The vehicle priorities express the vehicle's need of charging or discharging

        The priorities are calculated as follows:
        - First we find the area included between the max and min curves (max/min area)
        - The charging priority is calculated as the ratio of
            - the area above the y = current_charge line that is part of the "max/min area"
            - and the "max/min area"
        - The discharging priority is calculated as the ratio of
            - the area below the y = current_charge line that is part of the "max/min area"
            - and the "max/min area"
        - If area is equal to zero and next_max_charge = next_max_discharge = 0 then both priorities are equal to 0

        From the above it is obvious that the following is true for the two priorities:
        ``charging_priority = 1 - discharging_priority``
        """
        print('Tracing update_priorities')
        # x_axes = tf.range(13.0)
        x_axes = tf.range(self._time_before_departure + 1.0)
        max_curve, min_curve = self._calculate_charge_curves()

        max_curve_area = tfp.math.trapz(max_curve)
        min_curve_area = tfp.math.trapz(min_curve)
        diff_curve_area = max_curve_area - min_curve_area
        max_intersection = tf_find_intersection_v2(x_axes,
                                                   max_curve,
                                                   self._current_charge)

        min_intersection = tf.cond(tf.less(0.0, max_intersection),
                                   lambda: 0.0,
                                   lambda: tf_find_intersection_v2(x_axes,
                                                           min_curve,
                                                           self._current_charge))

        if max_intersection > 0.0:
            intersection = max_intersection
            curve = max_curve - self._current_charge
        else:
            intersection = min_intersection
            curve = self._current_charge - min_curve
        ceil = tf.math.ceil(intersection)
        curve = tf.pad(curve[:tf.cast(ceil, tf.int32)],
                       [[0, 1]],
                       constant_values=0.0)
        x_values = tf.pad(x_axes[:tf.cast(ceil, tf.int32)],
                          [[0, 1]],
                          constant_values=intersection)
        curve_area = tfp.math.trapz(curve, x_values)
        pr = tf.cond(tf.not_equal(intersection, 0.0),
                     lambda: tf.math.divide_no_nan(curve_area, diff_curve_area),
                     lambda: tf.cond(tf.less_equal(self._current_charge, self._target_charge),
                                     lambda: 0.0,
                                     lambda: 1.0))
        pr_c = tf.cond(tf.math.logical_and(tf.equal(pr, 0.0), tf.equal(self._current_charge, self._target_charge)),
                       lambda: 0.0,
                       lambda: 1.0 - pr)

        pr /= tf.math.maximum(1.0, self._time_before_departure - 2.0)
        pr_c /= tf.math.maximum(1.0, self._time_before_departure - 2.0)
        pr = my_round(pr, tf.constant(2))
        pr_c = my_round(pr_c, tf.constant(2))
        if max_intersection > 0.0:
            self._charge_priority.assign(pr)
            self._discharge_priority.assign(pr_c)
        else:
            self._charge_priority.assign(pr_c)
            self._discharge_priority.assign(pr)

    @tf.function
    def update(self):
        """
        Update state variables
        """
        print('Tracing update')
        self._update_next_charging_states()
        self._update_priorities()

    @tf.function
    def update_current_charge(self,
                              charging_coefficient: tf.Tensor,
                              normalization_constant: tf.Tensor,
                              residue_energy: tf.Tensor):
        """
        Update current charge by providing:
            - the total energy, gained or lost, divided by the number of cars in the parking
            - the mean charge/discharge priority
            - any residue energy that wasn't allocated by the previous vehicles

        The current charge is updated based on the following formula:
            ``current_charge = current_charge + energy *
            (1 + charge/discharge_priority - normalization_constant) + residue_energy``

        ### Arguments:
            energy (``float``) :
                description: The total energy, bought or sold in this timestep, divided by the number
                of cars in the parking
            normalization_constant (``float``) :
                description: The normalization constant to keep the sum of added/subtract energy equal to the total
            residue_energy (``float``) :
                description: Any residue energy that wasn't allocated by the previous vehicles

        ### Returns:
            float : The residue energy that wasn't allocated by this vehicle
        """
        print('Tracing update_current_charge')
        is_charging = tf.less_equal(0.0, charging_coefficient)
        priority = tf.cond(is_charging,
                           lambda: self._charge_priority,
                           lambda: self._discharge_priority)
        next_max_energy = tf.cond(is_charging,
                           lambda: self._next_max_charge,
                           lambda: self._next_max_discharge)
        sign = tf.cond(is_charging,
                       lambda: tf.constant(1.0),
                       lambda: tf.constant(-1.0))

        new_vehicle_energy = (
            next_max_energy * charging_coefficient * (1.0 + priority - normalization_constant) + residue_energy
        )
        residue = tf.math.maximum(0.0, tf.math.abs(new_vehicle_energy) - next_max_energy) * sign

        new_charge = my_round(self._current_charge + new_vehicle_energy - residue, tf.constant(2), )
        self._current_charge.assign(new_charge)
        self._time_before_departure.assign_sub(1.0)

        # if self._time_before_departure != 0:
        #     self.update()
        tf.cond(tf.not_equal(self._time_before_departure, 0.0),
                lambda: self.update(),
                lambda: tf.no_op())

        return residue

    @tf.function
    def update_emergency_demand(self):
        """
        Satisfy the minimum demand of the vehicle
        """
        print('Tracing update_emergency_demand')
        new_current_charge = self._current_charge + self._next_min_charge - self._next_min_discharge
        self._current_charge.assign(new_current_charge)
        self._next_max_charge.assign_sub(self._next_min_charge)
        self._next_max_discharge.assign_sub(self._next_min_discharge)
        self._next_min_charge.assign(0.0)
        self._next_min_discharge.assign(0.0)

    @tf.function
    def get_current_charge(self):
        """
        Get current charge

        ### Returns:
            float : The current battery's charge
        """
        return tf.constant(self._current_charge)

    @tf.function
    def get_target_charge(self):
        """
        Get the target charge

        ### Returns:
            float : The desired charge to be achieved before departure
        """
        return tf.constant(self._target_charge)

    @tf.function
    def get_time_before_departure(self):
        """
        Get the total time before departure

        ### Returns:
            int : The total time remaining before departure
        """
        return tf.cast(self._time_before_departure, tf.int32)

    @tf.function
    def get_max_charge(self):
        """
        Get max charge

        ### Returns:
            float : The maximum allowed charge for the vehicle's battery
        """
        return self._max_charge

    @tf.function
    def get_min_charge(self):
        """
        Get min charge

        ### Returns:
            float : The minimum allowed charge for the vehicle's battery
        """
        return self._min_charge

    @tf.function
    def get_next_max_charge(self):
        """
        Get next max charge

        ### Returns:
            float : The next maximum charge that can be achieved without compromising the target
        """
        return tf.constant(self._next_max_charge)

    @tf.function
    def get_next_min_charge(self):
        """
        Get next min charge

        ### Returns:
            float : The next minimum charge that can be achieved without compromising the target
        """
        return tf.constant(self._next_min_charge)

    @tf.function
    def get_next_max_discharge(self):
        """
        Get next max discharge

        ### Returns:
            float : The next maximum discharge that can be achieved without compromising the target
        """
        return tf.constant(self._next_max_discharge)

    @tf.function
    def get_next_min_discharge(self):
        """
        Get next min discharge

        ### Returns:
            float : The next minimum discharge that can be achieved without compromising the target
        """
        return tf.constant(self._next_min_discharge)

    @tf.function
    def get_charge_priority(self):
        """
        Get charge priority

        ### Returns:
            float : The charge priority of the vehicle
        """
        return tf.constant(self._charge_priority)

    @tf.function
    def get_discharge_priority(self):
        """
        Get discharge priority

        ### Returns:
            float : The discharge priority of the vehicle
        """
        return tf.constant(self._discharge_priority)
    
    def toJson(self) -> Dict[str, Any]:
        return {
            # "class": Vehicle.__name__,
            # "_name": self._name,
            "current_change": self._current_charge,
            "target_charge": self._target_charge,
            # "original_target_charge": self._original_target_charge,
            "time_before_departure": self._time_before_departure,
            # "max_charge": self._max_charge,
            # "min_charge": self._min_charge,
            # "max_charging_rate": self._max_charging_rate,
            # "min_discharging_rate": self._max_discharging_rate,
            "next_max_charge": self._next_max_charge,
            "next_min_charge": self._next_min_charge,
            "next_max_discharge": self._next_max_discharge,
            "next_min_discharge": self._next_min_discharge,
            "charge_priority": self._charge_priority,
            "discharge_priority": self._discharge_priority,
        }

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
