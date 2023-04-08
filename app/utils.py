import math
import random
from itertools import cycle
from typing import List, Tuple, Union, Optional, Callable

import gin
import numpy as np
from intersect import intersection
# from tensorflow.keras import layers
# from tensorflow.keras.models import load_model
# from tf_agents.agents.dqn.dqn_agent import DdqnAgent
# from tf_agents.networks import sequential
import tensorflow as tf

from app.models.tf_utils import my_round, my_round_vectorized


def find_intersection(x1, y1, x2, y2) -> Union[Tuple[float, float], None]:
    x, y = intersection(x1, y1, x2, y2)
    intersections = list(filter(lambda val: val[0] > 0, set(zip(x, y))))

    if len(intersections) == 0:
        return None

    return intersections[0]


def find_intersection_v2(x1, y1, c) -> Union[Tuple[tf.Tensor, tf.Tensor], None]:
    length = len(y1)
    for i in range(length - 2, -1, -1):
        if y1[i + 1] == c:
            return x1[i + 1], c
        elif y1[i] > c > y1[i + 1] or y1[i] < c < y1[i + 1]:
            y = abs(y1[i] - y1[i + 1])
            y_ = abs(c - y1[i + 1])
            x = abs(x1[i + 1] - x1[i])
            x_ = (1 - y_ / y) * x
            return x1[i] + x_, c

    return None

class VehicleDistributionList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_vehicles_list = [0] * 24
        if len(self) > 0: self.compute_avg_vehicles_list()

    def compute_avg_vehicles_list(self):
        total_days = len(self) // 24
        if total_days <= 0:
            return
        avg_vehicles_list = [0.0] * 24
        for i in range(len(self) // 24 * 24):
            for v in self[i]:
                for z in range(v[2]):
                    avg_vehicles_list[(i + 1) % 24 + z - 1] += 1
        self.avg_vehicles_list = [x / total_days for x in avg_vehicles_list]





def create_vehicle_distribution(steps: object = 24 * 30 * 6, offset: int = 0, coefficient_function = None) -> List[List[int]]:
    if steps % 24 != 0:
        raise Exception('Invalid steps, need be multiple of 24')
    time_of_day = 0
    # day_coefficient = math.sin(math.pi / 6 * time_of_day + offset) / 2 + 0.5
    coefficient_calculator = lambda x: math.sin(math.pi / 6 * x + offset) / 2 + 0.5 if coefficient_function == None \
                                                                                    else coefficient_function
    vehicles = VehicleDistributionList([])
    for _ in range(steps):
        # day_coefficient = math.sin(math.pi / 6 * time_of_day + offset) / 2 + 0.5
        day_coefficient = coefficient_calculator(time_of_day)
        new_cars = max(0, int(np.random.normal(20 * day_coefficient, 2 * day_coefficient)))
        if time_of_day > 21 and time_of_day < 24:
            vehicles.append([])
        else:
            vehicles.append(
                [
                    (   min(
                            24 - time_of_day % 24,  # think it's better, allows to have vehicles at 23:00
                            random.randint(7, 12),
                        ),
                        round(6 + random.random() * 20, 2),
                        round(34 + random.random() * 20, 2),
                    )
                    for _ in range(new_cars)
                ]
            )
        time_of_day += 1
        time_of_day %= 24

    # TODO fix compute of _avg_vehicles_list
    vehicles.compute_avg_vehicles_list()

    return vehicles

# def create_dqn_agent(model_dir: str, agent_name: Optional[str] = None, num_actions = 21, **kwargs,) -> DdqnAgent:
#     if not model_dir:
#         layers_list = \
#             [
#                 layers.Dense(units=34, activation="elu"),
#                 layers.Dense(units=128, activation="elu"),
#                 layers.BatchNormalization(),
#                 layers.Dense(
#                     num_actions,
#                     activation=None,
#                 ),
#             ]
#     else:
#         keras_model = load_model(model_dir)
#         layers_list = []
#         i = 0
#         while True:
#             try:
#                 layers_list.append(keras_model.get_layer(index=i))
#             except IndexError:
#                 print('{0}: Total number of layers in neural network: {1}'.format(agent_name, i))
#                 break
#             except ValueError:
#                 print('{0}: Total number of layers in neural network: {1}'.format(agent_name, i))
#                 break
#             else:
#                 i += 1
#
#     q_net = sequential.Sequential(layers_list)
#     return DdqnAgent(
#         q_network=q_net,
#         **kwargs,
#     )

def compute_avg_vehicle_list(coefficient_function: Callable, num_of_days: int = 100,):
    generator = vehicle_arrival_generator(coefficient_function=coefficient_function, vehicle_list=None)
    if not isinstance(num_of_days, int):
        raise Exception('num_of_days entered is not integer')
    if num_of_days <= 0:
        return
    avg_vehicles_list = [0.0] * 24
    for i in range(num_of_days):
        for d in range(24):
            for v in next(generator):
                for z in range(v[0]):
                    avg_vehicles_list[(i + 1) % 24 + z - 1] += 1
    return [x / num_of_days for x in avg_vehicles_list]

def vehicle_arrival_generator_for_tf(coefficient_function: Callable, vehicles_list: List, train: bool = True):
    if train and not coefficient_function or not train and not vehicles_list:
        raise Exception("Improper intialization of vehicle generator. Either call the train version with a "
                        "coefficient function or the eval (train = False) with a vehicles_list")
    def eval_vehicle_generator():
        l = len(vehicles_list)
        vehicles_tensor = tf.ragged.constant(vehicles_list, tf.float32)
        index = tf.constant(0)
        while True:
            tensor = vehicles_tensor[index].to_tensor()
            tensor = tf.cond(tf.less(0, tf.shape(tensor)[0]),
                             lambda: tf.concat((tensor[..., 1:2], tensor[..., 2:3], tensor[..., 0:1]), axis=1),
                             lambda: tf.zeros((0,3), tf.float32)
                             )
            yield tensor
            index += 1
            index %= l
    def train_vehicle_generator():
        time_of_day = tf.constant(0)
        while True:
            day_coefficient = coefficient_function(tf.cast(time_of_day, tf.float32))
            new_cars = tf.cond(tf.less(time_of_day, 22),
                               lambda: tf.math.floor(tf.maximum(0.0,
                                                           tf.random.normal([],
                                                                            10.0 * day_coefficient,
                                                                            2.0 * day_coefficient),
                                                          ),
                                               ),
                               lambda: tf.constant(0.0))
            departure = lambda: tf.cast(tf.minimum(24 - time_of_day % 24,
                                                   tf.random.uniform((), 7, 13, tf.int32)),
                                        tf.float32)
            current_charge = lambda: my_round(6.0 + tf.random.uniform(()) * 20.0, tf.constant(2))
            target_charge = lambda: my_round(6.0 + tf.random.uniform(()) * 34.0, tf.constant(2))
            c = lambda i, t: i < new_cars
            b = lambda i, t: (i + 1.0,
                              tf.concat((t, tf.expand_dims(tf.stack((current_charge(), target_charge(), departure())),
                                                           axis=0)),
                                        axis=0))
            s = (0.0, tf.zeros((0, 3), tf.float32))
            i, vehicles = tf.while_loop(c,
                                        b,
                                        s,
                                        shape_invariants=(tf.TensorSpec((), tf.float32),
                                                          tf.TensorSpec((None, 3), tf.float32)))

            yield vehicles
            time_of_day += 1
            time_of_day %= 24

    return train_vehicle_generator if train else eval_vehicle_generator

def vehicle_arrival_generator(coefficient_function: Optional[Callable],
                            vehicle_list: Optional[Union[VehicleDistributionList, List]]):
    """
    Generator of vehicles according to given vehicle distribution (it just gives next element)
    or generated through a normal distribution parametrised by a given coefficient function
    """
    if bool(coefficient_function) == bool(vehicle_list):
        raise ValueError('Incompatible arguments given. Generator either '
                         'produces values using coeffiecient_function or iterates'
                         ' through given vehicle list not both')

    time_of_day = 0

    if bool(coefficient_function):
        while True:
            day_coefficient = coefficient_function(time_of_day)
            if time_of_day > 21:
                new_cars = 0
            else:
                new_cars = max(0, int(np.random.normal(10 * day_coefficient, 2 * day_coefficient)))
            vehicles = [
                (
                    min(
                        24 - time_of_day % 24,  # think it's better, allows to have vehicles at 23:00
                        random.randint(7, 12),
                    ),
                    round(6 + random.random() * 20, 2),
                    round(34 + random.random() * 20, 2),
                )
                for _ in range(new_cars)
            ]
            yield vehicles , time_of_day
            time_of_day += 1
            time_of_day %= 24

    else:
        vehicle_cycle = cycle(vehicle_list)
        while True:
            yield next(vehicle_cycle), time_of_day
            time_of_day += 1
            time_of_day %= 24

def tf_vehicle_generator(coefficient_function: Optional[Callable],
                         vehicle_list: Optional[Union[VehicleDistributionList, List]]):
    if coefficient_function and vehicle_list:
        raise Exception('Cannot create a generator with both coefficient'
                        'function and vehicle_list')
    if vehicle_list:
        vehicle_ragged_tensor = tf.ragged.constant(vehicle_list,
                                                   dtype=tf.float32,
                                                   )

    def vehicle_generator(time_of_day):
        day_coefficient = coefficient_function(tf.cast(time_of_day, tf.float32))
        new_cars = tf.cond(tf.less(time_of_day, 22),
                           lambda: tf.math.floor(tf.maximum(0.0,
                                                       tf.random.normal([],
                                                                        10.0 * day_coefficient,
                                                                        2.0 * day_coefficient),
                                                      ),
                                           ),
                           lambda: tf.constant(0.0))
        vehicles = tf.zeros(shape=(new_cars,))
        vehicles = tf.map_fn(lambda t: tf.convert_to_tensor([tf.cast(tf.minimum(24 - time_of_day % 24,
                                                                                tf.random.uniform((), 7, 12, tf.int32)),
                                                                     dtype=tf.float32),
                                                             my_round(6.0 + tf.random.uniform((), maxval=1.0) * 20.0,
                                                                      tf.constant(2)),
                                                             my_round(34.0 + tf.random.uniform((), maxval=1.0) * 20.0,
                                                                      tf.constant(2))]),
                             vehicles,
                             dtype=tf.float32)
        return vehicles

    def vehicle_cycle(index):
        index = index % len(vehicle_list)
        return vehicle_ragged_tensor[index]

    if coefficient_function:
        return vehicle_generator
    else:
        return vehicle_cycle


def generate_vehicles(coefficient_function):
    def create_vehicles(time_of_day):
        # print('Tracing create_vehicles')
        day_coefficient = coefficient_function(tf.cast(time_of_day, tf.float32))
        new_cars = tf.cond(tf.less(time_of_day, 22),
                           lambda: tf.math.floor(tf.maximum(0.0,
                                                            tf.random.normal([],
                                                                             10.0 * day_coefficient,
                                                                             2.0 * day_coefficient),
                                                            ),
                                                 ),
                           lambda: tf.constant(0.0))
        new_cars = tf.cast(new_cars, tf.int32)
        random_current_charge = my_round_vectorized(6.0 + tf.random.uniform((new_cars,)) * 20.0,
                                                    tf.constant(2))
        random_target_charge = my_round_vectorized(6.0 + tf.random.uniform((new_cars,)) * 34.0,
                                                   tf.constant(2))
        random_departures = tf.cast(tf.minimum(24 - tf.fill((new_cars,), time_of_day) % 24,
                                               tf.random.uniform((new_cars,), 7, 13, tf.int32)),
                                    tf.float32)
        return tf.transpose(tf.stack((random_current_charge,
                                      random_target_charge,
                                      random_departures),
                                     axis=0))

        # departure = lambda: tf.cast(tf.minimum(24 - time_of_day % 24,
        #                                        tf.random.uniform((), 7, 13, tf.int32)),
        #                             tf.float32)
        # current_charge = lambda: my_round(6.0 + tf.random.uniform(()) * 20.0, tf.constant(2))
        # target_charge = lambda: my_round(6.0 + tf.random.uniform(()) * 34.0, tf.constant(2))
        # c = lambda i, t: i < new_cars
        # b = lambda i, t: (i + 1.0,
        #                   tf.concat((t, tf.expand_dims(tf.stack((current_charge(), target_charge(), departure())),
        #                                                axis=0)),
        #                             axis=0))
        # s = (0.0, tf.zeros((0, 3), tf.float32))
        # i, vehicles = tf.while_loop(c,
        #                             b,
        #                             s,
        #                             shape_invariants=(tf.TensorSpec((), tf.float32),
        #                                               tf.TensorSpec((None, 3), tf.float32)))
        # return vehicles

    return create_vehicles

def calculate_vehicle_avg_distribution_from_tensor(days: int = 100,
                                                   generator = None):
    if generator is None:
        generator = generate_vehicles(lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5)
    t = tf.constant([[1.0] + [0.0] * 23])
    @tf.function
    def loop():
        result = tf.zeros((0, 24))
        for i in tf.range(days):
            tf.autograph.experimental.set_loop_options(maximum_iterations=days,
                                                       shape_invariants=[(result, tf.TensorShape([None, 24]))])
            for d in tf.range(24):
                cars = generator(d)
                cars_added = tf.repeat(t, tf.shape(cars)[0], axis=0)
                update_indexes = tf.concat((tf.reshape(tf.range(tf.shape(cars_added)[0]), [-1, 1]),
                                            tf.cast(cars[..., 2:3], tf.int32)),
                                           axis=1)
                cars_added = tf.tensor_scatter_nd_update(cars_added,
                                                         update_indexes,
                                                         (-1.0) * tf.ones_like(cars[..., 2]))
                cars_added = tf.cumsum(cars_added, axis=1)
                cars_added = tf.roll(cars_added, d, axis=1)
                result = tf.concat((result, cars_added), axis=0)
        return tf.reduce_sum(result, axis=0)
                # for c in cars:


                # for c in cars:
                #     ones = tf.ones([c[2]], tf.float32)
                #     padding = [[d, 24 - d - tf.cast(c[2], tf.int32)]]
                #     ones = tf.pad(ones, padding, constant_values=0.0)
                #     tensor += ones

    return (loop() / tf.cast(days, tf.float32))


# def compute_avg_consuption_list(list, generator):
#     if bool(list) == bool(generator):
#         raise Exception(" Can't have both a list and a generator, only pick one")
#     final_list = [0.0] * 24
#     if list:
#         days = len(list)
#         for day in list:
#