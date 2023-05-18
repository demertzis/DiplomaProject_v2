import numpy as np
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
import tensorflow as tf
import app.models.tf_vehicle_8 as Vehicle
import app.models.tf_parking_6 as Parking
import app.models.tf_vehicle_7 as Vehicle_2
import time

# tf.config.run_functions_eagerly(True)
num_of_vehicles = 200
vehicles = tf.zeros((num_of_vehicles, 13), tf.float32)
# vehicles_2 = tf.zeros((num_of_vehicles, 13), tf.float16)

# parking = Parking.Parking(num_of_vehicles, 'asdf')

update = tf.function(jit_compile=True)(Vehicle._update_priorities)
update_2 = tf.function(jit_compile=True)(Vehicle_2._update_priorities)
# update = tf.function(jit_compile=False)(parking._sort_vehicles_for_charge_update)
# update_2 = tf.function(jit_compile=False)(parking._sort_vehicles_for_charge_update_2)

@tf.function(jit_compile=True)
def _calculate_charge_curves(vehicles: tf.Tensor, num_of_vehicles: int):
    #print('Tracing vectorized_calculate_charge_curves_2')
    unrolled_tensor = tf.concat((tf.range(12.0, -0.1, -1.0), tf.zeros((12,), tf.float32)), axis=0)
    t = tf.vectorized_map(lambda t: tf.roll(unrolled_tensor, t - 12, axis=0)[0:13],
                            tf.cast(vehicles[..., 2], tf.int32))
    temp = tf.repeat(tf.expand_dims(tf.range(13.0), 0), num_of_vehicles, axis=0)
    multiplier_tensor = tf.stack((temp,
                                    t),
                                   axis=0)
    max_curve_charge_rates = tf.stack((vehicles[..., 7:8], vehicles[..., 8:9]), axis=0)
    charge_rates = tf.stack((tf.reverse(max_curve_charge_rates, axis=[0]),
                             max_curve_charge_rates),
                            axis=0)
    max_curve_starting_values = tf.stack((vehicles[..., 0:1], vehicles[..., 1:2]), axis=0) * [[[[-1.0]]], [[[1.0]]]]
    curves_no_clip = multiplier_tensor * charge_rates + max_curve_starting_values
    curves_no_clip = tf.math.reduce_min(curves_no_clip, axis=1) * [[[-1.0]], [[1.0]]]
    return tf.clip_by_value(curves_no_clip,
                            clip_value_min=vehicles[..., 4:5],
                            clip_value_max=vehicles[..., 3:4])

@tf.function(jit_compile=True)
def _calculate_charge_curves_2(vehicles: tf.Tensor, num_of_vehicles: int):
    #print('Tracing vectorized_calculate_charge_curves_2')
    temp = tf.repeat(tf.expand_dims(tf.range(13.0), 0), num_of_vehicles, axis=0)
    t = tf.where(tf.equal(0.0, temp - vehicles[..., 2:3]), 1.0, 0.0)
    t = tf.math.cumsum(t, axis=1, reverse=True)
    multiplier_tensor = tf.stack((tf.math.cumsum(t, axis=1, exclusive=True),
                                  tf.math.cumsum(t, axis=1, exclusive=True, reverse=True)),
                                 axis=0)
    max_curve_charge_rates = tf.stack((vehicles[..., 7:8], vehicles[..., 8:9]), axis=0)
    min_curve_charge_rates = tf.reverse(max_curve_charge_rates, axis=[0])
    max_curve_starting_values = tf.stack((vehicles[..., 0:1], vehicles[..., 1:2]), axis=0)
    max_curve_no_clip = multiplier_tensor * max_curve_charge_rates + max_curve_starting_values
    min_curve_no_clip = multiplier_tensor * min_curve_charge_rates - max_curve_starting_values
    max_clip = vehicles[..., 3:4]
    min_clip = vehicles[..., 4:5]
    max_curve_no_clip = tf.minimum(max_curve_no_clip[0], max_curve_no_clip[1])
    min_curve_no_clip = tf.minimum(min_curve_no_clip[0], min_curve_no_clip[1])
    # min_curve_no_clip = tf.maximum(min_curve_no_clip[0], min_curve_no_clip[1])
    return tf.clip_by_value(max_curve_no_clip,
                            clip_value_min=min_clip,
                            clip_value_max=max_clip),\
           tf.clip_by_value(min_curve_no_clip,
                            clip_value_min= min_clip,
                            clip_value_max= max_clip)


_calculate_charge_curves(vehicles, num_of_vehicles)
_calculate_charge_curves_2(vehicles, num_of_vehicles)
st = time.time()
for _ in range(1000):
    _calculate_charge_curves(vehicles, num_of_vehicles)
print(time.time() - st)
for _ in range(1000):
    _calculate_charge_curves(vehicles, num_of_vehicles)
print(time.time() - st)