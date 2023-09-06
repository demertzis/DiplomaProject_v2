import tensorflow as tf
# import math
# from app.models.tf_parking_single_agent import Parking
# from app.utils import generate_vehicles_single_model as generator
# car_gen = generator([lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5,
#                      lambda x: tf.math.sin(math.pi / 6.0 * (x + 3.0)) / 2.0 + 0.5],
#                     [4,2],
#                     100
#                     )
# t = car_gen(tf.constant(12, tf.int64))
# parking = Parking(100, 3, 'A')
# parking.assign_vehicles(t)
# print('')

t = tf.constant(1)
t = 2*t