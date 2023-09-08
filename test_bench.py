import json
import math

import tensorflow as tf

from app.models.tf_parking_single_agent import Parking
from app.utils import generate_vehicles_single_model as generator, VehicleDistributionListConstantShape
from config import VEHICLE_BATTERY_CAPACITY, VEHICLE_MIN_CHARGE

tf.config.run_functions_eagerly(True)

with open('data/vehicles_constant_shape.json') as file:
    vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))

car_gen = generator([lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5,
                     lambda x: tf.math.sin(math.pi / 6.0 * (x + 3.0)) / 2.0 + 0.5],
                    [4, 2],
                    100
                    )
# t = car_gen(tf.constant(12, tf.int64))
t = tf.repeat([tf.constant(vehicles[0])], 6, axis=0)
max_min_charges = tf.repeat(tf.expand_dims(tf.repeat([[VEHICLE_BATTERY_CAPACITY, VEHICLE_MIN_CHARGE] + [0.0] * 8],
                                                     100,
                                                     axis=0), axis=0),
                            6,
                            axis=0)
parking = Parking(100, 6, 'A')

parking.assign_vehicles(tf.concat((t, max_min_charges), axis=-1))
f = parking.return_fields()
print('')
