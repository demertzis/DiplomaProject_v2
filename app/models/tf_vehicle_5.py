from tf_agents.typing import types
from tf_agents.utils import common

from app.models.tf_utils import tf_find_intersection_v2, my_pad
import json
from typing import Any, Dict, NamedTuple
import config
import tensorflow as tf
import tensorflow_probability as tfp

from app.models.tf_utils import my_round

# class Vehicle:
#     """
#     A class to represent a vehicle and its charging functionalities
# 
#     ### Arguments:
#         initial_charge (``float``) :
#             description: The initial charge of the vehicle's battery
#         target_charge (``float``) :
#             description: The desired charge to be achieved in `total_stay` hours
#         total_stay (``int``) :
#             description: The total hours that the vehicle will remain in the parking
#         max_charge (``float``) :
#             description: The maximum allowed charge to be stored in the vehicle's battery
#         min_charge (``float``) :
#             description: The minimum allowed charge to be stored in the vehicle's battery
# 
#     ### Attributes:
#         _charge_priority (``float``) :
#             description: The charge priority of the vehicle
#         _discharge_priority (``float``) :
#             description: The discharge priority of the vehicle
#         _next_max_charge (``float``) :
#             description: The maximum charging state needed to be achieved in the next hour so tha
#             the target_charge is achievable
#         _next_min_charge (``float``) :
#             description: The minimum charging state needed to be achieved in the next hour so that
#             the target_charge is achievable
#     """
@tf.function
def park(max_charging_rate: tf.Tensor, max_discharging_rate: tf.Tensor, vehicle: tf.Tensor):
    """
    Park vehicle to parking space and initialize its state variables
    To be called by the parking space it was assigned to

    ### Arguments:
        max_charging_rate (``float``):
            description: The maximum charging rate
        max_discharging_rate (``float``):
            description: The maximum discharging rate
    """
    #print('Tracing park')
    v = tf.tensor_scatter_nd_update(vehicle, [[7], [8]], [max_charging_rate, max_discharging_rate])
    return update(v)

@tf.function
def _calculate_next_max_charge(current_charge: tf.Tensor,
                               time_before_departure: tf.Tensor,
                               vehicle: tf.Tensor):
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
    #print('Tracing calculate_next_max_charge')
    return my_round(
        tf.math.reduce_min(
            tf.stack((vehicle[7] + current_charge,
                      vehicle[3],
                      tf.clip_by_value((time_before_departure - 1.0),
                                       0.0,
                                       12.0) * \
                      vehicle[8] + \
                      vehicle[1],),
                      axis=0)
            ),
        tf.constant(3),
        )

@tf.function
def _calculate_next_min_charge(current_charge: tf.Tensor,
                               time_before_departure: tf.Tensor,
                               vehicle: tf.Tensor):
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
    #print('Tracing calculate_next_min_charge')
    return my_round(
        tf.math.reduce_max(
            tf.stack(
                (current_charge - vehicle[8],
                 vehicle[4],
                 vehicle[1] - \
                 tf.clip_by_value((time_before_departure - 1.0),
                                  0.0,
                                  11.0) * \
                 vehicle[7],),
                axis=0),
        ),
        tf.constant(3),
    )

@tf.function 
def _update_next_charging_states(vehicle: tf.Tensor):
    """
    Update max and min charge state variables
    - ``ΔΕ+(max) = max(0, next_max_charge - current_charge)``
    - ``ΔΕ+(min) = max(0, next_min_charge - current_charge)``
    - ``ΔΕ-(max) = max(0, current_charge - next_min_charge)``
    - ``ΔΕ-(min) = max(0, current_charge - next_max_charge)``
    """
    #print('Tracing update_next_charging_states')
    next_max_charge = _calculate_next_max_charge(vehicle[0],
                                                 vehicle[2],
                                                 vehicle)
    next_min_charge = _calculate_next_min_charge(vehicle[0],
                                                 vehicle[2],
                                                 vehicle)
    new_max_charge = my_round(tf.math.maximum(0.0, next_max_charge - vehicle[0]), tf.constant(3))
    new_min_charge = my_round(tf.math.maximum(0.0, next_min_charge - vehicle[0]), tf.constant(3))
    new_max_discharge = my_round(tf.math.maximum(0.0, vehicle[0] - next_min_charge), tf.constant(3))
    new_min_discharge = my_round(tf.math.maximum(0.0, vehicle[0] - next_max_charge), tf.constant(3))
    return tf.tensor_scatter_nd_update(vehicle,
                                       [[9], [10], [11], [12]],
                                       [new_max_charge,
                                        new_min_charge,
                                        new_max_discharge,
                                        new_min_discharge])
@tf.function
def _calculate_charge_curves(vehicle: tf.Tensor):
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
    #print('Tracing calculate_charge_curves')
    current_max_charge = vehicle[0]
    # current_min_charge = current_max_charge
    time_before_departure = tf.cast(vehicle[2], tf.int32)
    # max_charges_stacked = tf.zeros((13,))
    # min_charges_stacked = tf.zeros((13,))
    charges_stacked = tf.tensor_scatter_nd_update(tf.zeros((2, 13)), [[0, 0], [1, 0]], [current_max_charge,
                                                                                        current_max_charge])
    upper_limit = time_before_departure + 1
    c = lambda i, charges_stacked: i < 13
    b = lambda i, charges_stacked: (i + 1,
                                    tf.tensor_scatter_nd_update(charges_stacked,
                                                                tf.convert_to_tensor([[0, i], [1, i]]),
                                                                [_calculate_next_max_charge(charges_stacked[0, i-1],
                                                                                            tf.cast(upper_limit - i,
                                                                                                    tf.float32),
                                                                                            vehicle),
                                                                 _calculate_next_min_charge(charges_stacked[1, i-1],
                                                                                            tf.cast(upper_limit - i,
                                                                                                    tf.float32),
                                                                                            vehicle),]))
    last_index, charges_stacked = tf.while_loop(c,
                                                b,
                                                [1, charges_stacked],)
    # charges_stacked = tf.tensor_scatter_nd_update(charges_stacked, tf.stack(tf.range))
    return charges_stacked[0], charges_stacked[1]
    # c = lambda i, cmxc, cmnc: i > 0.0
    # b = lambda i, cmxc, cmnc: (i - 1.0,
    #                            tf.concat((cmxc,
    #                                       tf.expand_dims(_calculate_next_max_charge(cmxc[-1],
    #                                                                                 i,
    #                                                                                 vehicle), axis=0)),
    #                                      0),
    #                            tf.concat((cmnc,
    #                                       tf.expand_dims(_calculate_next_min_charge(cmnc[-1],
    #                                                                                 i,
    #                                                                                 vehicle), axis=0)),
    #                                      0))
    # _,\
    # max_charges_stacked,\
    # min_charges_stacked = tf.while_loop(c,
    #                                     b,
    #                                     [time_before_departure, current_max_charge, current_min_charge],
    #                                     shape_invariants=[time_before_departure.get_shape(),
    #                                                       tf.TensorShape([None]),
    #                                                       tf.TensorShape([None])],
    #                                     parallel_iterations=3)
    # size = tf.shape(min_charges_stacked)[0]
    # min_charges_stacked = tf.pad(min_charges_stacked,
    #                              [[0, 13 - size]],
    #                              constant_values=min_charges_stacked[-1])
    # min_charges_stacked = tf.ensure_shape(min_charges_stacked, [13,])
    # max_charges_stacked = tf.pad(max_charges_stacked,
    #                              [[0, 13 - size]],
    #                              constant_values=max_charges_stacked[-1])
    # max_charges_stacked = tf.ensure_shape(max_charges_stacked, [13, ])
    # return max_charges_stacked, min_charges_stacked

@tf.function
def _update_priorities(vehicle: tf.Tensor):
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
    #print('Tracing update_priorities')
    x_axes = tf.range(13.0)
    # x_axes = tf.range(vehicle[2] + 1.0)
    max_curve, min_curve = _calculate_charge_curves(vehicle)
    current_charge = vehicle[0]
    target_charge = vehicle[1]
    time_before_departure = vehicle[2]

    max_curve_area = tfp.math.trapz(max_curve)
    min_curve_area = tfp.math.trapz(min_curve)
    diff_curve_area = max_curve_area - min_curve_area
    max_intersection = tf_find_intersection_v2(x_axes,
                                               max_curve,
                                               current_charge)
    # min_intersection = tf.cond(tf.less(0.0, max_intersection),
    #                            lambda: 0.0,
    #                            lambda: tf_find_intersection_v2(x_axes,
    #                                                            min_curve,
    #                                                            current_charge))
    min_intersection = tf_find_intersection_v2(x_axes,
                                               min_curve,
                                               current_charge)
    # if max_intersection > 0.0:
    #     intersection = max_intersection
    #     curve = max_curve - current_charge
    # else:
    #     intersection = min_intersection
    #     curve = current_charge - min_curve
    intersection = tf.maximum(max_intersection, min_intersection)
    curve_1 = tf.clip_by_value(max_curve - current_charge,
                               0.0,
                               60.0)
    curve_2 = tf.clip_by_value(current_charge - min_curve,
                               0.0,
                               60.0)
    curve = tf.cond(tf.less(0.0, max_intersection),
                            lambda: curve_1,
                            lambda: curve_2)
    return curve
    x_values = tf.tensor_scatter_nd_update(x_axes,
                                           [[tf.cast(tf.math.ceil(intersection), tf.int32)]],
                                           [intersection])
    curve_area = tfp.math.trapz(curve, x_values)
    # ceil = tf.math.ceil(intersection)
    # curve = tf.pad(curve[:tf.cast(ceil, tf.int32)],
    #                [[0, 1]],
    #                constant_values=0.0)
    # x_values = tf.pad(x_axes[:tf.cast(ceil, tf.int32)],
    #                   [[0, 1]],
    #                   constant_values=intersection)
    # curve_area = tfp.math.trapz(curve, x_values)
    pr = tf.cond(tf.not_equal(intersection, 0.0),
                 lambda: tf.math.divide_no_nan(curve_area, diff_curve_area),
                 lambda: tf.cond(tf.less_equal(current_charge, target_charge),
                                 lambda: 0.0,
                                 lambda: 1.0))
    pr_c = tf.cond(tf.math.logical_and(tf.equal(pr, 0.0), tf.equal(current_charge, target_charge)),
                   lambda: 0.0,
                   lambda: 1.0 - pr)

    pr /= tf.math.maximum(1.0, time_before_departure - 2.0)
    pr_c /= tf.math.maximum(1.0, time_before_departure - 2.0)
    pr = my_round(pr, tf.constant(2))
    pr_c = my_round(pr_c, tf.constant(2))
    # if max_intersection > 0.0:
    #     return tf.tensor_scatter_nd_update(vehicle, [[5], [6]], [pr, pr_c])
    # else:
    #     return tf.tensor_scatter_nd_update(vehicle, [[5], [6]], [pr_c, pr])
    return_tensor = tf.cond(tf.less(0.0, max_intersection),
                            lambda: tf.tensor_scatter_nd_update(vehicle, [[5], [6]], [pr, pr_c]),
                            lambda: tf.tensor_scatter_nd_update(vehicle, [[5], [6]], [pr_c, pr]))
    return return_tensor

@tf.function
def update(vehicle: tf.Tensor):
    """
    Update state variables
    """
    #print('Tracing update')
    v = _update_next_charging_states(vehicle)
    return _update_priorities(v)

@tf.function
def update_current_charge(charging_coefficient: tf.Tensor,
                          normalization_constant: tf.Tensor,
                          residue_energy: tf.Tensor,
                          vehicle: tf.Tensor):
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
    #print('Tracing update_current_charge')
    # is_charging = tf.less_equal(0.0, charging_coefficient)
    # priority = tf.cond(is_charging,
    #                    lambda: vehicle[5],
    #                    lambda: vehicle[6])
    # next_max_energy = tf.cond(is_charging,
    #                    lambda: vehicle[9],
    #                    lambda: vehicle[11])
    # sign = tf.cond(is_charging,
    #                lambda: tf.constant(1.0),
    #                lambda: tf.constant(-1.0))
    priority, next_max_energy, sign = tf.cond(tf.less_equal(0.0, charging_coefficient),
                                              lambda: (vehicle[5], vehicle[9], tf.constant(1.0)),
                                              lambda: (vehicle[6], vehicle[11], tf.constant(-1.0)),)
    new_vehicle_energy = (
        next_max_energy * charging_coefficient * (1.0 + priority - normalization_constant) + residue_energy
    )
    residue = tf.math.maximum(0.0, tf.math.abs(new_vehicle_energy) - next_max_energy) * sign

    new_charge = my_round(vehicle[0] + new_vehicle_energy - residue, tf.constant(2),)

    new_vehicle = tf.tensor_scatter_nd_update(vehicle, [[0], [2]], [new_charge, vehicle[2] - 1.0])
    new_vehicle = tf.cond(tf.not_equal(new_vehicle[2], 0.0),
                          lambda: update(new_vehicle),
                          lambda: new_vehicle)

    return residue, new_vehicle

@tf.function
def update_emergency_demand(vehicle: tf.Tensor):
    """
    Satisfy the minimum demand of the vehicle
    """
    #print('Tracing update_emergency_demand')
    new_current_charge = vehicle[0] + vehicle[10] - vehicle[12]
    return tf.tensor_scatter_nd_update(vehicle,
                                       [[0], [9], [11], [10], [12]],
                                       [new_current_charge,
                                        vehicle[9] - vehicle[10],
                                        vehicle[11] - vehicle[12],
                                        tf.constant(0.0, tf.float32),
                                        tf.constant(0.0, tf.float32)])
#
# def toJson(self) -> Dict[str, Any]:
#     return {
#         # "class": __name__,
#         # "_name": self._name,
#         "current_change": self._current_charge,
#         "target_charge": self._target_charge,
#         # "original_target_charge": self._original_target_charge,
#         "time_before_departure": self._time_before_departure,
#         # "max_charge": self._max_charge,
#         # "min_charge": self._min_charge,
#         # "max_charging_rate": self._max_charging_rate,
#         # "min_discharging_rate": self._max_discharging_rate,
#         "next_max_charge": self._next_max_charge,
#         "next_min_charge": self._next_min_charge,
#         "next_max_discharge": self._next_max_discharge,
#         "next_min_discharge": self._next_min_discharge,
#         "charge_priority": self._charge_priority,
#         "discharge_priority": self._discharge_priority,
#     }
#
# def __repr__(self) -> str:
#     return json.dumps(self.toJson(), indent=4)
