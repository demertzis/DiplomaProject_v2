import tensorflow as tf
from tf_agents.specs import tensor_spec, array_spec
import time
import math
import numpy as np
from tf_agents.specs.array_spec import sample_spec_nest
from tf_agents.trajectories.time_step import TimeStep

from app.models.tf_vehicle_4 import VehicleFields
from app.models.tf_parking_3 import Parking
# from app.models.tf_vehicle_3 import Vehicle
# from app.models.tf_parking import Parking
from app.models.parking import Parking as Parking_2
from app.models.vehicle import Vehicle
from app.utils import vehicle_arrival_generator

from tf_agents.utils.nest_utils import assert_matching_dtypes_and_inner_shapes
# tf.config.run_functions_eagerly(True)


# parking = Parking(100, 'train')
# parking_2 = Parking_2(100, 'train')
# vehicle_generator = VehicleArrivalGenerator(lambda x: math.sin(math.pi / 6 * x) / 2 + 0.5, None)
#
# def parkings_update():
#     coefficient = tf.constant(np.random.uniform(low=-1.0, high=1.0, size=()), dtype=tf.float32)
#     parking.update(coefficient)
#     parking_2.update(coefficient.numpy())
#
# def print_simmilarities():
#     if tf.shape(parking._vehicles)[0] == 0 or len(parking_2._vehicles) == 0:
#         return
#     v1 = parking._vehicles[0]
#     v1 = VehicleFields(*tf.unstack(v1))
#     v2 = parking_2._vehicles[0]
#     print('             TF-Parking       Old-Parking\n'
#           'num_vehicles     {}                 {}   \n'
#           'charge-co        {}                 {}   \n'
#           'discharge-co     {}                 {}   \n'
#           'current_charge   {}                 {}   \n'
#           'parking_charge   {}                 {}   \n'
#           'parking_mean_pr  {}                 {}   \n'
#           'parking_next_max {}                 {}   \n'.format(parking._num_of_vehicles.numpy(),
#                                                                parking_2.get_current_vehicles(),
#                                                                v1.charge_priority,
#                                                                v2.get_charge_priority(),
#                                                                v1.discharge_priority,
#                                                                v2.get_discharge_priority(),
#                                                                v1.current_charge,
#                                                                v2.get_current_charge(),
#                                                                parking.get_current_energy(),
#                                                                parking_2.get_current_energy(),
#                                                                parking.get_charge_mean_priority(),
#                                                                parking_2.get_charge_mean_priority(),
#                                                                parking.get_next_max_charge(),
#                                                                parking_2.get_next_max_charge(),))
# def update_till_departure():
#     while parking._num_of_vehicles.numpy() and len(parking_2._vehicles):
#         st = time.time()
#         parkings_update()
#         et = time.time()
#         print('Elapsed time:', et - st, ' seconds')
#         print_simmilarities()
#
# tst = time.time()
# total_cars = 0
# for i in range(20):
#
#     new_cars, _ = next(vehicle_generator)
#     total_cars += len(new_cars)
#     for total_stay, initial_charge, target_charge in new_cars:
#         v = VehicleFields(initial_charge, target_charge, total_stay, 60.0, 0.0)
#         v_2 = Vehicle(initial_charge, target_charge, total_stay, 60.0, 0.0)
#         parking.assign_vehicle(v)
#         parking_2.assign_vehicle(v_2)
#     update_till_departure()
# tet = time.time()
# print('Total Time: ', tet - tst, 'seconds')


#         # tf.print(self.__dict__
#
#     @tf.function
#     def static_method(self):
#         print("Tracing")
#         # tf.print(self.field)
#
#         return Test(2 * self.field, 2 * self.field_2)
#
#
# i = Test(tf.constant(4.0))
# y = Test()
# i = i.static_method()
# st = time.time()
# y = i.static_method()
# y = y.static_method()
# et = time.time()
# print(et - st)
# print(i)
# print(y)

specs = tensor_spec.from_spec(TimeStep(
    step_type=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
    discount=array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
    reward=array_spec.ArraySpec(shape=(), dtype=np.float32),
    observation=array_spec.BoundedArraySpec(shape=(34,), dtype=np.float32, minimum=-1., maximum=1.),
))

@tf.function
def tf_test():
    i = tf.constant(True)
    while i:
        time_step = sample_spec_nest(specs, )
        if assert_matching_dtypes_and_inner_shapes(
                time_step,
                specs,
                allow_extra_fields=True,
                caller='asf',
                tensors_name="`experience`",
                specs_name="`train_argspec`"):
            continue
        else:
            tf.print(assert_matching_dtypes_and_inner_shapes(
                time_step,
                specs,
                allow_extra_fields=True,
                caller='asf',
                tensors_name="`experience`",
                specs_name="`train_argspec`"))
            break

tf_test()



