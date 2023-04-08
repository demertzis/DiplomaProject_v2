from tf_agents.typing import types
from tf_agents.utils import common

from app.models.tf_utils import tf_find_intersection_vectorized, my_pad
import json
from typing import Any, Dict, NamedTuple
import config
import tensorflow as tf
import tensorflow_probability as tfp

from app.models.tf_utils import my_round_vectorized
@tf.function(input_signature=[tf.TensorSpec([], tf.float32),
                              tf.TensorSpec([], tf.float32),
                              tf.TensorSpec([None, 13], tf.float32)])
def park(max_charging_rate: tf.Tensor, max_discharging_rate: tf.Tensor, vehicles: tf.Tensor):
    """
    Park vehicle to parking space and initialize its state variables
    To be called by the parking space it was assigned to

    ### Arguments:
        max_charging_rate (``float``):
            description: The maximum charging rate
        max_discharging_rate (``float``):
            description: The maximum discharging rate
    """
    print('Tracing vectorized_park')
    c = tf.convert_to_tensor([[0., 0., 0., 0., 0., 0., 0., max_charging_rate, max_discharging_rate, 0., 0., 0., 0.]])
    new_vehicles = vehicles + tf.tile(c, [tf.shape(vehicles)[0], 1])
    return update(new_vehicles)

@tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name='Current_Charge'),
                              tf.TensorSpec([None], tf.float32, name='Time_Before_Departure'),
                              tf.TensorSpec([None, 13], tf.float32, name='Vehicles')])
def _calculate_next_max_charge(current_charge: tf.Tensor,
                               time_before_departure: tf.Tensor,
                               vehicles: tf.Tensor):
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
    final_tensor = tf.math.reduce_min(tf.stack((vehicles[..., 7] + current_charge,
                                                vehicles[..., 3],
                                                tf.clip_by_value((time_before_departure - 1.0),
                                                                 0.0,
                                                                 12.0) * \
                                                vehicles[..., 8] + \
                                                vehicles[..., 1],)),
                                      axis=0)
    return my_round_vectorized(final_tensor, tf.constant(3))


@tf.function(input_signature=[tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([None, 13], tf.float32)])
def _calculate_next_min_charge(current_charge: tf.Tensor,
                               time_before_departure: tf.Tensor,
                               vehicles: tf.Tensor):
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
    final_tensor = tf.math.reduce_max(tf.stack((current_charge - vehicles[..., 8],
                                                vehicles[..., 4],
                                                vehicles[..., 1] - \
                                                    tf.clip_by_value((time_before_departure - 1.0),
                                                                      0.0,
                                                                      12.0) * \
                                                    vehicles[..., 7]),
                                               axis=0),
                                      axis=0)
    return my_round_vectorized(final_tensor, tf.constant(3))

@tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32)])
def _update_next_charging_states(vehicles: tf.Tensor):
    """
    Update max and min charge state variables
    - ``ΔΕ+(max) = max(0, next_max_charge - current_charge)``
    - ``ΔΕ+(min) = max(0, next_min_charge - current_charge)``
    - ``ΔΕ-(max) = max(0, current_charge - next_min_charge)``
    - ``ΔΕ-(min) = max(0, current_charge - next_max_charge)``
    """
    print('Tracing vectorized_update_next_charging_states')
    next_max_charge = _calculate_next_max_charge(vehicles[..., 0],
                                                 vehicles[..., 2],
                                                 vehicles)
    next_min_charge = _calculate_next_min_charge(vehicles[..., 0],
                                                 vehicles[..., 2],
                                                 vehicles)
    zeros = tf.zeros_like(next_max_charge)
    new_max_charge = my_round_vectorized(tf.math.maximum(zeros, next_max_charge - vehicles[..., 0]), tf.constant(3))
    new_min_charge = my_round_vectorized(tf.math.maximum(zeros, next_min_charge - vehicles[..., 0]), tf.constant(3))
    new_max_discharge = my_round_vectorized(tf.math.maximum(zeros, vehicles[..., 0] - next_min_charge), tf.constant(3))
    new_min_discharge = my_round_vectorized(tf.math.maximum(zeros, vehicles[..., 0] - next_max_charge), tf.constant(3))
    new_stack = tf.transpose(tf.stack((new_max_charge,
                                       new_min_charge,
                                       new_max_discharge,
                                       new_min_discharge,),
                                      axis=0))
    return tf.concat((vehicles[..., :9], new_stack), axis=1)

@tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32)])
def _calculate_charge_curves(vehicles: tf.Tensor):
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
    print('Tracing vectorized_calculate_charge_curves_2')
    num_vehicles = tf.shape(vehicles)[0]
    time_before_departure = vehicles[..., 2:3]
    tbd_indexes = tf.concat((tf.expand_dims(tf.range(num_vehicles), axis=1),
                             tf.cast(time_before_departure, tf.int32)),
                            axis=1)
    zeros = tf.zeros((num_vehicles, 13), tf.float32)
    t = tf.tensor_scatter_nd_update(zeros, tbd_indexes, tf.ones_like(vehicles[..., 2]))
    t = tf.math.cumsum(t, axis=1, reverse=True)
    multiplier_tensor = tf.concat((tf.math.cumsum(t, exclusive=True, axis=1),
                                   tf.math.cumsum(t, axis=1, exclusive=True, reverse=True)),
                                  axis=1)
    multiplier_tensor = tf.reshape(multiplier_tensor, [-1, 13])
    charge_rates = vehicles[..., 7:9]
    max_curve_charge_rates = tf.reshape(charge_rates, [-1, 1])
    min_curve_charge_rates = tf.reshape(tf.reverse(charge_rates, axis=[-1]), [-1, 1])
    max_curve_starting_values = tf.reshape(vehicles[..., 0:2], [-1, 1])
    max_curve_no_clip = multiplier_tensor * max_curve_charge_rates + max_curve_starting_values
    min_curve_no_clip = multiplier_tensor * min_curve_charge_rates - max_curve_starting_values
    clip_tensor = vehicles[..., 3:5] * [[1., -1.]]
    max_curve_no_clip = tf.minimum(max_curve_no_clip[0::2], max_curve_no_clip[1::2])
    min_curve_no_clip = tf.minimum(min_curve_no_clip[0::2], min_curve_no_clip[1::2])
    return tf.clip_by_value(max_curve_no_clip,
                            clip_value_min=clip_tensor[..., 1:2],
                            clip_value_max=clip_tensor[..., 0:1]),\
        tf.clip_by_value(min_curve_no_clip,
                         clip_value_min=(-1.0)*clip_tensor[..., 0:1],
                         clip_value_max=clip_tensor[..., 1:2]) * (-1.0)

@tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32)])
def _update_priorities(vehicles: tf.Tensor):
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
    print('Tracing vectorized_update_priorities')
    x_axes = tf.range(13.0)
    max_curve, min_curve = _calculate_charge_curves(vehicles)
    current_charge = vehicles[..., 0]
    target_charge = vehicles[..., 1]
    time_before_departure = vehicles[..., 2]

    curves = tf.concat((min_curve, max_curve), axis=1)
    curves = tf.reshape(curves, [-1, 13])
    curves_areas = tfp.math.trapz(curves)
    diff_curve_areas = curves_areas[1::2] - curves_areas[::2]
    current_charge_doubled = tf.expand_dims(tf.repeat(current_charge, tf.fill(tf.shape(current_charge), 2), axis=0),
                                            axis=1)
    intersections = tf_find_intersection_vectorized(x_axes,
                                                    curves,
                                                    current_charge_doubled)
    intersections = tf.reshape(intersections, [-1, 2])
    intersection_gather_index = tf.expand_dims(tf.math.argmax(intersections, axis=1, output_type=tf.int32), axis=1)
    num_vehicles = tf.shape(vehicles)[0]
    vehicle_indices = tf.expand_dims(tf.range(num_vehicles), axis=1)
    intersection_gather_index = tf.concat((vehicle_indices,
                                           intersection_gather_index),
                                          axis=1)
    all_curves = tf.clip_by_value((curves - current_charge_doubled) * tf.tile([[-1.0], [1.0]], [num_vehicles, 1]),
                                  0.0,
                                  60.0)
    all_curves = tf.reshape(all_curves, [-1, 2, 13])
    chosen_curves = tf.gather_nd(all_curves, intersection_gather_index)

    chosen_intersections = tf.gather_nd(intersections, intersection_gather_index)
    intersectins_ceil = tf.concat((vehicle_indices,
                                   tf.cast(tf.math.ceil(tf.expand_dims(chosen_intersections, axis=1)),
                                           tf.int32)),
                                  axis=1)
    x_values = tf.tensor_scatter_nd_update(tf.tile(tf.expand_dims(x_axes, 0),
                                                   [num_vehicles, 1]),
                                           intersectins_ceil,
                                           chosen_intersections)
    curve_areas = tfp.math.trapz(chosen_curves, x_values)
    where_x = tf.math.divide_no_nan(curve_areas, diff_curve_areas)
    where_y = tf.where(tf.less_equal(current_charge, target_charge),
                       tf.zeros_like(current_charge, tf.float32),
                       tf.ones_like(current_charge, tf.float32))
    pr = tf.where(tf.less(0.0, chosen_intersections), where_x, where_y)
    pr_c = tf.where(tf.math.logical_or(tf.not_equal(pr, 0.0), tf.not_equal(current_charge, target_charge)),
                    1.0 - pr,
                    tf.zeros_like(current_charge, tf.float32))

    pr = my_round_vectorized(pr / tf.math.maximum(1.0, time_before_departure - 2.0),
                             tf.constant(2))
    pr_c = my_round_vectorized(pr_c / tf.math.maximum(1.0, time_before_departure - 2.0),
                             tf.constant(2))
    probs_stacked = tf.stack((pr, pr_c), axis=0)
    final_probs_update = tf.where(tf.less(0, intersection_gather_index[..., 1:]),
                                  tf.transpose(probs_stacked),
                                  tf.transpose(tf.reverse(probs_stacked, [0])))
    return tf.concat((vehicles[..., :5], final_probs_update, vehicles[..., 7:]), axis=1)

@tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32)])
def update(vehicle: tf.Tensor):
    """
    Update state variables
    """
    print('Tracing vectorized_update')
    v = _update_next_charging_states(vehicle)
    return _update_priorities(v)

@tf.function(input_signature=[tf.TensorSpec([], tf.float32),
                              tf.TensorSpec([], tf.float32),
                              tf.TensorSpec([None, 13], tf.float32)])
def update_current_charge(charging_coefficient: tf.Tensor,
                          normalization_constant: tf.Tensor,
                          vehicles: tf.Tensor):
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
    print('Tracing vectorized_update_current_charge')
    priority, next_max_energy, sign = tf.cond(tf.less_equal(0.0, charging_coefficient),
                                              lambda: (vehicles[..., 5], vehicles[..., 9], tf.constant(1.0)),
                                              lambda: (vehicles[..., 6], vehicles[..., 11], tf.constant(-1.0)),)
    new_vehicle_energy = (next_max_energy * \
                          charging_coefficient * \
                          (1.0 + priority - normalization_constant)) * \
                         sign
    surplus_tensor = new_vehicle_energy - next_max_energy
    residue_cumsum = tf.cumsum(surplus_tensor)
    final_residue = tf.where(tf.less(residue_cumsum, surplus_tensor), surplus_tensor, residue_cumsum)
    negative_max_energy = tf.math.reduce_max(next_max_energy) * (-1.0)
    final_energy = tf.clip_by_value(final_residue, negative_max_energy, 0.0) + next_max_energy

    new_energy = my_round_vectorized(vehicles[..., 0] + sign * final_energy, tf.constant(2))
    new_vehicles = tf.concat((tf.expand_dims(new_energy, axis=1),
                              vehicles[..., 1:2],
                              vehicles[..., 2:3] - 1.0,
                              vehicles[..., 3:]),
                             axis=1)

    return update(new_vehicles)

@tf.function(input_signature=[tf.TensorSpec([None, 13], tf.float32)])
def update_emergency_demand(vehicles: tf.Tensor):
    """
    Satisfy the minimum demand of the vehicle
    """
    print('Tracing update_emergency_demand')
    template = vehicles[..., 10:11]
    zeros = tf.zeros_like(template)
    return tf.concat((vehicles[..., 0:1] + template - vehicles[..., 12:13],
                      vehicles[..., 1:9],
                      vehicles[..., 9:10] - template,
                      zeros,
                      vehicles[..., 11:12] - vehicles[..., 12:13],
                      zeros),
                     axis=1)

