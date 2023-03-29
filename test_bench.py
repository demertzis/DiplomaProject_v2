import json

import tensorflow as tf
from tf_agents.specs import tensor_spec, array_spec
import time
import math
import numpy as np
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories.time_step import TimeStep

from app.error_handling import ParkingIsFull
from app.models.tf_vehicle_4 import VehicleFields
from app.models.tf_vehicle_4 import Vehicle as Vehicle_4
# from app.models.tf_vehicle_5 import Vehicle as Vehicle_5
import app.models.tf_vehicle_5 as Vehicle_5
import app.models.tf_vehicle_6 as Vehicle_6
from app.models.tf_parking_3 import Parking
from app.models.tf_parking_4 import Parking as Parking_3
# from app.models.tf_vehicle_3 import Vehicle
# from app.models.tf_parking import Parking
from app.models.parking import Parking as Parking_2
from app.models.vehicle import Vehicle
from app.utils import vehicle_arrival_generator,\
    VehicleDistributionList,\
    vehicle_arrival_generator_for_tf,\
    generate_vehicles

# tf.config.run_functions_eagerly(True)
iterations = 10

parking = Parking(200, 'train')
parking_2 = Parking_2(200, 'train')
parking_3 = Parking_3(200, 'train')

with open('data/vehicles.json') as file:
    vehicles = VehicleDistributionList(json.load(file))

vehicle_generator = vehicle_arrival_generator(None, list(vehicles))
vehicles_tensor = tf.ragged.constant(list(vehicles))

def parkings_update():
    coefficient = tf.constant(np.random.uniform(low=-1.0, high=1.0, size=()), dtype=tf.float32)
    print('coefficient: ', coefficient.numpy())
    start = time.time()
    parking.update(coefficient)
    # parking.update(tf.constant(0.5))
    end = time.time()
    print('tf_parking', end - start)
    start = time.time()
    parking_2.update(coefficient.numpy())
    # parking_2.update(tf.constant(0.5).numpy())
    end = time.time()
    print('python_parking', end - start)
    start = time.time()
    parking_3.update(coefficient)
    # parking_3.update(tf.constant(0.5))
    end = time.time()
    print('tf_new_parking', end - start)
    print_simmilarities()


def print_simmilarities():
    if tf.shape(parking_3._vehicles)[0] == 0 or len(parking_2._vehicles) == 0:
        return
    # v1 = parking._vehicles[0]
    # v1 = VehicleFields(*tf.unstack(v1))
    # v2 = parking_2._vehicles[0]
    print('             TF-Parking       Old-Parking\n'
          'num_vehicles     {}                 {}   \n'
          # 'charge-co        {}                 {}   \n'
          # 'discharge-co     {}                 {}   \n'
          # 'current_charge   {}                 {}   \n'
          'parking_charge   {}                 {}   \n'
          'parking_mean_pr  {}                 {}   \n'
          'parking_next_max {}                 {}   \n'.format(parking_3._num_of_vehicles.numpy(),
                                                               parking_2.get_current_vehicles(),
                                                               # v1.charge_priority,
                                                               # v2.get_charge_priority(),
                                                               # v1.discharge_priority,
                                                               # v2.get_discharge_priority(),
                                                               # v1.current_charge,
                                                               # v2.get_current_charge(),
                                                               parking_3.get_current_energy(),
                                                               parking_2.get_current_energy(),
                                                               parking_3.get_charge_mean_priority(),
                                                               parking_2.get_charge_mean_priority(),
                                                               parking_3.get_next_max_charge(),
                                                               parking_2.get_next_max_charge(),))

gen = vehicle_arrival_generator_for_tf(None, list(vehicles), False)
# for i in gen():
#     print(i)
# tf_generator = tf.data.Dataset.from_generator(vehicle_arrival_generator_for_tf(lambda x:
#                                                                                       tf.math.sin(math.pi / 6.0 * x) /
#                                                                                       2.0 + 0.5),
#                                               output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32)).prefetch(10)
tf_generator = tf.data.Dataset.from_generator(vehicle_arrival_generator_for_tf(None, list(vehicles), False),
                                              output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32)).prefetch(10)
# tf_generator
# vehicle_dataset = tf_generator.map(lambda item: tf.cond(tf.equal(tf.shape(item)[0], 0),
#                                                         lambda: item,
#                                                         lambda: tf.pad(item, [[0,0], [0, 1]], constant_values=60.0),
#                                                         ),
#                                    num_parallel_calls=tf.data.AUTOTUNE,
#                                   )
#
# vehicle_dataset = vehicle_dataset.map(lambda item: tf.cond(tf.equal(tf.shape(item)[0], 0),
#                                                         lambda: item,
#                                                         lambda: tf.pad(item, [[0,0], [0, 9]], constant_values=0.0),
#                                                         ),
#                                    num_parallel_calls=tf.data.AUTOTUNE,
#                                   ).cache()

vehicle_iterator = iter(tf_generator)
# vehicle_iterator = tf_generator.get_single_element()
@tf.function
def tf_new_park():
    print('Tracing tf_new_park')
    c = tf.constant(60.0)
    vehicles = next(vehicle_iterator)
    # vehicles = tf.gather(vehicle_iterator, [[index]])
    num_of_vehicles = tf.shape(vehicles)[0]
    if num_of_vehicles == 0:
        return tf.zeros(shape=(0, 0))
    else:
        max_min_charges = tf.repeat([[60., 0.]], repeats=tf.shape(vehicles)[0], axis=0)
        vehicles = tf.concat((vehicles, max_min_charges), axis=1)
        vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
        parked_tensor = tf.map_fn(lambda t: tf.cond(tf.equal(tf.shape(t)[0], 0),
                                                    lambda: t,
                                                    lambda: Vehicle_5.park(c, c, t)),
                                  vehicles,
                                  parallel_iterations=10)
        return parked_tensor

vehicle_computer = Vehicle_4()
# vehicle_computer_2 = Vehicle_5()
# i = 0
# for _ in range(iterations):
#     new_cars, _ = next(vehicle_generator)
#     print(len(new_cars))
#     st = time.time()
#     for total_stay, initial_charge, target_charge in new_cars:
#         v = VehicleFields(initial_charge, target_charge, tf.cast(total_stay, tf.float32), 60.0, 0.0)
#         # parking.assign_vehicle(v)
#         vehicle_computer.assign_fields(v)
#         vehicle_computer.park(60.0, 60.0)
#         vehicle = vehicle_computer.return_fields()
#     print('tf_parking_park: ', time.time() - st)
#     st = time.time()
#     for total_stay, initial_charge, target_charge in new_cars:
#         v = VehicleFields(initial_charge, target_charge, tf.cast(total_stay, tf.float32), 60.0, 0.0)
#         v_2 = Vehicle(initial_charge, target_charge, total_stay, 60.0, 0.0)
#         v_2.park(60.0, 60.0)
#     print('py_parking_park: ', time.time() - st)
#     st = time.time()
#     v_3 = tf_new_park()
#     # v_3 = tf_new_park(i)
#     # i += 1
#     print('new_tf_parking_park: ', time.time() - st)
#     print('')


# def update_till_departure():
#     while parking._num_of_vehicles.numpy() and len(parking_2._vehicles):
#         # st = time.time()
#         parkings_update()
#         # et = time.time()
#         # #print('Elapsed time:', et - st, ' seconds')
#         print_simmilarities()

@tf.function
def tf_parking_update():
    vehicles = next(vehicle_iterator)
    max_min_charges = tf.repeat([[60., 0.]], repeats=tf.shape(vehicles)[0], axis=0)
    vehicles = tf.concat((vehicles, max_min_charges), axis=1)
    vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
    if tf.shape(vehicles)[1] == 10:
        tf.print("Flag")
    parking_3.assign_vehicles(vehicles)
tst = time.time()
total_cars = 0
strategy = tf.distribute.get_strategy()  # Default strategy that works on CPU and single GPU
#print('Running on CPU instead')

with strategy.scope():
    t = tf.constant(0)
    for i in range(iterations):
        # new_cars, _ = next(vehicle_generator)
        # total_cars += len
        vehicles = next(vehicle_iterator)
        max_min_charges = tf.repeat([[60., 0.]], repeats=tf.shape(vehicles)[0], axis=0)
        vehicles = tf.concat((vehicles, max_min_charges), axis=1)
        vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
        # parking_3.assign_vehicles(vehicles)
        st = time.time()
        # for total_stay, initial_charge, target_charge in new_cars:
        #     v = VehicleFields(initial_charge, target_charge, tf.cast(total_stay, tf.float32), 60.0, 0.0)
        #     parking.assign_vehicle(v)
        for v in vehicles:
            v = VehicleFields(*tf.unstack(v))
            parking.assign_vehicle(v)
        print('tf_parking: ', time.time() - st)

        st = time.time()
        # for total_stay, initial_charge, target_charge in new_cars:
            # v = VehicleFields(initial_charge, target_charge, tf.cast(total_stay, tf.float32), 60.0, 0.0)
            # v_2 = Vehicle(initial_charge, target_charge, total_stay, 60.0, 0.0)
            # try:
            #     parking_2.assign_vehicle(v_2)
            # except ParkingIsFull:
            #     print("Parking is full no more cars added")
        for v in vehicles:
            v_2 = Vehicle(v[0].numpy(), v[1].numpy(), int(v[2].numpy()), 60.0, 0.0)
            try:
                parking_2.assign_vehicle(v_2)
            except ParkingIsFull:
                print("Parking is full no more cars added")

        print('py_parking: ', time.time() - st)

        st = time.time()
        # vehicles = tf.gather(vehicles_tensor, t).to_tensor()
        # vehicles = tf.cond(tf.less(0, tf.shape(vehicles)[0]),
        #                    lambda: tf.concat((vehicles[..., 1:2],
        #                                       vehicles[..., 2:3],
        #                                       vehicles[..., 0:1]), axis=1),
        #                    lambda: tf.zeros((0, 3), tf.float32)
        #                    )
        # # num_of_vehicles = tf.shape(vehicles)[0]
        # max_min_charges = tf.repeat([[60., 0.]], repeats=tf.shape(vehicles)[0], axis=0)
        # # if num_of_vehicles == 0:
        # #     vehicles = tf.zeros([0,13], tf.float32)
        # # else:
        # #     vehicles = tf.concat((vehicles, max_min_charges), axis=1)
        # #     vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
        # vehicles = tf.concat((vehicles, max_min_charges), axis=1)
        # vehicles = tf.pad(vehicles, [[0, 0], [0, 8]], constant_values=0.0)
        # t += 1
        # tf_parking_update()
        parking_3.assign_vehicles(vehicles)
        print('tf_new_parking: ', time.time() - st)
        parkings_update()

    # tet = time.time()
    # #print('Total Time: ', tet - tst, 'seconds')
    #     new_cars, _ = next(vehicle_generator)
    #     total_cars += len(new_cars)
    #     for total_stay, initial_charge, target_charge in new_cars:
    #         v = VehicleFields(initial_charge, target_charge, total_stay, 60.0, 0.0)
    #         v_2 = Vehicle(initial_charge, target_charge, total_stay, 60.0, 0.0)
    #         parking.assign_vehicle(v)
    #         parking_2.assign_vehicle(v_2)
    #         parking_3.assign_vehicle(v)
    #         st = time.time()
    #         parking_2._update_parking_state()
    #         print('python_parking: ', time.time() - st)
    #         st = time.time()
    #         parking._update_parking_state()
    #         print('old_parking: ', time.time() - st)
    #         st = time.time()
    #         parking_3._update_parking_state()
    #         print('new_parking: ', time.time() - st)

#
# @tf.function
# def tf_test():
#     t = tf.zeros(shape=(100, 100))
#     for i in tf.range(100):
#         t = t + 1
#     tf.print(t)
#
# t = tf.Variable(tf.zeros([100, 100], tf.int32))
# @tf.function
# def tf_v_test():
#     for i in tf.range(100):
#         t.assign_add(tf.ones([100, 100], tf.int32))
#     tf.print(t)
#
# def py_test():
#     t = [[0] * 100] * 100
#     for i in range(100):
#         t = [[x + 1 for x in row] for row in t]
#     print(t)
#
# def np_test():
#     t = np.zeros(shape=(100,100))
#     for i in range(100):
#         t = t + 1
#     print(t)
#
# tf_test()
# st = time.time()
# tf_test()
# print('tf-time: ', time.time() - st)
#
# tf_v_test()
# st = time.time()
# tf_v_test()
# print('tf-variable-time: ', time.time() - st)
#
# st = time.time()
# py_test()
# print('py-time: ', time.time() - st)
#
# st = time.time()
# np_test()
# print('np-time: ', time.time() - st)
