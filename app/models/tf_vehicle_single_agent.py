import tensorflow as tf
import tensorflow_probability as tfp

from app.models.tf_utils import tf_find_intersection_vectorized_single_model


@tf.function(jit_compile=True)
def park(vehicles: tf.Tensor, max_charging_rate, max_discharging_rate, num_of_vehicles: int, num_of_agents: int):
    """
    Park vehicle to parking space and initialize its state variables
    To be called by the parking space it was assigned to

    ### Arguments:
        max_charging_rate (``float``):
            description: The maximum charging rate
        max_discharging_rate (``float``):
            description: The maximum discharging rate
    """
    # print('Tracing vectorized_park')
    mult = tf.constant([[1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., ]],
                       tf.float32)
    c = tf.constant([[0., 0., 0., 0., 0., 0., 0., max_charging_rate, max_discharging_rate, 0., 0., 0., 0.]],
                    tf.float32)
    new_vehicles = vehicles * mult + c
    correct_mult = tf.clip_by_value(new_vehicles[..., 2:3],
                                    0.0,
                                    1.0)
    return update(new_vehicles * correct_mult, num_of_vehicles, num_of_agents)


def _calculate_next_max_charge(vehicles: tf.Tensor):
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
    # print('Tracing calculate_next_max_charge')
    final_tensor = tf.math.reduce_min(tf.stack((vehicles[..., 7] + vehicles[..., 0],
                                                vehicles[..., 3],
                                                tf.clip_by_value((vehicles[..., 2] - 1.0),
                                                                 0.0,
                                                                 12.0) * \
                                                vehicles[..., 8] + \
                                                vehicles[..., 1],)),
                                      axis=0)
    return final_tensor


def _calculate_next_min_charge(vehicles: tf.Tensor):
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
    # print('Tracing calculate_next_min_charge')
    final_tensor = tf.math.reduce_max(tf.stack((vehicles[..., 0] - vehicles[..., 8],
                                                vehicles[..., 4],
                                                vehicles[..., 1] - \
                                                tf.clip_by_value((vehicles[..., 2] - 1.0),
                                                                 0.0,
                                                                 12.0) * \
                                                vehicles[..., 7]),
                                               axis=0),
                                      axis=0)
    return final_tensor


def _update_next_charging_states(vehicles: tf.Tensor):
    """
    Update max and min charge state variables
    - ``ΔΕ+(max) = max(0, next_max_charge - current_charge)``
    - ``ΔΕ+(min) = max(0, next_min_charge - current_charge)``
    - ``ΔΕ-(max) = max(0, current_charge - next_min_charge)``
    - ``ΔΕ-(min) = max(0, current_charge - next_max_charge)``
    """
    # print('Tracing vectorized_update_next_charging_states')
    next_max_charge = _calculate_next_max_charge(vehicles)
    next_min_charge = _calculate_next_min_charge(vehicles)
    zeros = tf.zeros_like(next_max_charge, tf.float32)
    new_max_charge = tf.math.maximum(zeros, next_max_charge - vehicles[..., 0])
    new_min_charge = tf.math.maximum(zeros, next_min_charge - vehicles[..., 0])
    new_max_discharge = tf.math.maximum(zeros, vehicles[..., 0] - next_min_charge)
    new_min_discharge = tf.math.maximum(zeros, vehicles[..., 0] - next_max_charge)
    # new_stack = tf.transpose(tf.stack((new_max_charge,
    #                                    new_min_charge,
    #                                    new_max_discharge,
    #                                    new_min_discharge,),
    #                                   axis=0))
    new_stack = tf.stack((new_max_charge,
                          new_min_charge,
                          new_max_discharge,
                          new_min_discharge,),
                         axis=-1)
    return tf.concat((vehicles[..., :9], new_stack), axis=-1)


def _calculate_charge_curves(vehicles: tf.Tensor, num_of_vehicles: int, num_of_agents: int):
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
    # print('Tracing vectorized_calculate_charge_curves_2')
    unrolled_tensor = tf.concat((tf.range(12.0, -0.1, -1.0), tf.zeros((12,), tf.float32)), axis=0)
    # t = tf.vectorized_map(lambda t: tf.vectorized_map(lambda t_2: tf.roll(unrolled_tensor, t_2 - 12, axis=0)[0:13],
    #                                                   t),
    #                       tf.cast(vehicles[..., 2], tf.int32))
    t = tf.reshape(tf.vectorized_map(lambda t: tf.roll(unrolled_tensor, t - 12, axis=0)[0:13],
                                     tf.reshape((tf.cast(vehicles[..., 2], tf.int32)), shape=[-1])),
                   shape=[num_of_agents, num_of_vehicles, -1])
    temp = tf.repeat([tf.repeat(tf.expand_dims(tf.range(13.0), 0), num_of_vehicles, axis=0)], num_of_agents, axis=0)
    multiplier_tensor = tf.stack((temp,
                                  t),
                                 axis=0)
    max_curve_charge_rates = tf.stack((vehicles[..., 7:8], vehicles[..., 8:9]), axis=0)
    charge_rates = tf.stack((tf.reverse(max_curve_charge_rates, axis=[0]),
                             max_curve_charge_rates),
                            axis=0)
    # max_curve_starting_values = tf.stack((vehicles[..., 0:1], vehicles[..., 1:2]), axis=-3) * [[[[-1.0]]], [[[1.0]]]]
    max_curve_starting_values = tf.stack((vehicles[..., 0:1], vehicles[..., 1:2]), axis=0) * [[[[[-1.0]]]], [[[[1.0]]]]]
    curves_no_clip = multiplier_tensor * charge_rates + max_curve_starting_values
    curves_no_clip = tf.math.reduce_min(curves_no_clip, axis=1) * [[[[-1.0]]], [[[1.0]]]]
    return tf.clip_by_value(curves_no_clip,
                            # clip_value_min=vehicles[..., 4:5],
                            clip_value_min=vehicles[..., 4:5],
                            # clip_value_max=vehicles[..., 3:4])
                            clip_value_max=vehicles[..., 3:4])


# @tf.function
def _update_priorities(vehicles: tf.Tensor, num_of_vehicles: int, num_of_agents: int):
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
    # print('Tracing vectorized_update_priorities')
    x_axes = tf.range(13.0)
    curves = _calculate_charge_curves(vehicles, num_of_vehicles, num_of_agents)
    current_charge = vehicles[..., 0]
    target_charge = vehicles[..., 1]
    time_before_departure = vehicles[..., 2]
    curves_areas = tfp.math.trapz(curves)
    diff_curve_areas = curves_areas[1] - curves_areas[0]
    current_charge_expanded = tf.expand_dims(current_charge, axis=-1)
    intersections = tf_find_intersection_vectorized_single_model(x_axes,
                                                                 curves,
                                                                 current_charge_expanded,
                                                                 num_of_vehicles,
                                                                 num_of_agents)
    intersection_gather_index = tf.math.argmax(intersections, axis=0, output_type=tf.int32)
    coordinates = tf.meshgrid(tf.range(num_of_agents), tf.range(num_of_vehicles), indexing='ij')
    intersection_gather_index = tf.stack([intersection_gather_index] + coordinates, axis=-1)
    all_curves = tf.clip_by_value((curves - current_charge_expanded) * [[[[-1.0]]], [[[1.0]]]],
                                  0.0,
                                  60.0)
    chosen_curves = tf.gather_nd(all_curves, intersection_gather_index)
    chosen_intersections = tf.gather_nd(intersections, intersection_gather_index)
    intersection_ceil = tf.stack(coordinates +
                                 [tf.cast(tf.math.ceil(chosen_intersections), tf.int32)],
                                 axis=-1)
    total_x_axes = tf.repeat([tf.repeat([x_axes], num_of_vehicles, axis=0)], num_of_agents, axis=0)
    x_values = tf.tensor_scatter_nd_update(total_x_axes,
                                           intersection_ceil,
                                           chosen_intersections)

    curve_areas = tfp.math.trapz(chosen_curves, x_values)
    where_x = tf.math.divide_no_nan(curve_areas, diff_curve_areas)
    where_y = tf.cast(tf.less(target_charge, current_charge), tf.float32)
    pr = tf.cast(tf.less(0.0, chosen_intersections), tf.float32) * where_x + \
         tf.cast(tf.less_equal(chosen_intersections, 0.0), tf.float32) * where_y
    pr_c = tf.cast(tf.math.logical_or(tf.not_equal(pr, 0.0),
                                      tf.not_equal(current_charge, target_charge)),
                   tf.float32) * (1.0 - pr)
    pr = pr / tf.math.maximum(1.0, time_before_departure - 2.0)
    pr_c = pr_c / tf.math.maximum(1.0, time_before_departure - 2.0)
    probs_stacked = tf.stack((pr, pr_c), axis=-1)
    mask = tf.cast(tf.less(0, intersection_gather_index[..., :1]), tf.float32)
    final_probs_update = mask * probs_stacked + (1.0 - mask) * tf.reverse(probs_stacked, [-1])
    return tf.concat((vehicles[..., :5], final_probs_update, vehicles[..., 7:]), axis=-1)


# @tf.function
def update(vehicle: tf.Tensor, num_of_vehicles: int, num_of_agents: int):
    """
    Update state variables
    """
    # print('Tracing vectorized_update')
    v = _update_next_charging_states(vehicle)
    return _update_priorities(v, num_of_vehicles, num_of_agents)


# @tf.function(jit_compile=True)
def update_current_charge(charging_coefficient: tf.Tensor,
                          normalization_constant: tf.Tensor,
                          vehicles: tf.Tensor,
                          num_of_vehicles: int,
                          num_of_agents: int):
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
    # print('Tracing vectorized_update_current_charge')
    # condition = tf.less_equal(0.0, charging_coefficient)
    # condition = tf.sign(charging_coefficient)
    sign = tf.sign(charging_coefficient)
    neg_sign = tf.negative(sign)
    # priority = tf.where(condition, vehicles[..., 5], vehicles[..., 6])
    priority = tf.maximum(sign, 0.0) * vehicles[..., 5] + tf.maximum(neg_sign, 0.0) * vehicles[..., 6]
    # next_max_energy = tf.where(condition, vehicles[..., 9], vehicles[..., 11])
    next_max_energy = tf.maximum(sign, 0.0) * vehicles[..., 9] + tf.maximum(neg_sign, 0.0) * vehicles[..., 11]
    # sign = tf.where(condition, 1.0, -1.0)
    new_vehicle_energy = (next_max_energy * \
                          charging_coefficient * \
                          (1.0 + priority - normalization_constant)) * \
                         sign
    surplus_tensor = new_vehicle_energy - next_max_energy
    residue_cumsum = tf.cumsum(surplus_tensor)
    final_energy = tf.clip_by_value(residue_cumsum, tf.minimum(0.0, surplus_tensor), 0.0) + next_max_energy
    new_energy = vehicles[..., 0] + sign * final_energy
    # new_vehicles = tf.concat((tf.expand_dims(new_energy, axis=1),
    new_vehicles = tf.concat([tf.expand_dims(new_energy, axis=-1),
                              vehicles[..., 1:2],
                              tf.clip_by_value(vehicles[..., 2:3] - 1.0, 0.0, 24.0),
                              vehicles[..., 3:]],
                             axis=-1)
    return update(new_vehicles, num_of_vehicles, num_of_agents)


@tf.function(jit_compile=True)
def update_emergency_demand(vehicles: tf.Tensor):
    """
    Satisfy the minimum demand of the vehicle
    """
    # print('Tracing update_emergency_demand')
    min_charge = vehicles[..., 10:11]
    min_discharge = vehicles[..., 12:13]
    zeros = tf.zeros_like(min_charge)
    return tf.concat((vehicles[..., 0:1] + min_charge - min_discharge,
                      vehicles[..., 1:9],
                      vehicles[..., 9:10] - min_charge,
                      zeros,
                      vehicles[..., 11:12] - min_discharge,
                      zeros),
                     axis=-1)
