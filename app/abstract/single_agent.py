import math
import random
from typing import Callable, Optional, List

import tensorflow as tf
import numpy as np
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types
from tf_agents.agents import TFAgent, data_converter
from tf_agents.utils.common import Checkpointer
from tf_agents.utils.nest_utils import assert_matching_dtypes_and_inner_shapes

from app.models.tf_utils import my_round
from app.error_handling import ParkingIsFull
from app.models.tf_parking_3 import Parking
from app.models.tf_vehicle_4 import Vehicle, VehicleFields
from app.utils import vehicle_arrival_generator


# class MyAgentWrapper:
#     def __init__(self, agent: TFAgent, agent_id: int):
#         self._agent = agent
#         self._agent_id = agent_id - 1
#         self._preprocess_sequence_fn = common.function_in_tf1()(self._preprocess_sequence)
#
#     def _preprocess_sequence(self, experience: types.NestedTensor):
#         agent_reward = experience.reward[self._agent_id: self._agent_id + 1]
#         return experience.replace(reward=agent_reward)
#
#     def preprocess_sequence(self, experience: types.NestedTensor):
#         if self._validate_args:
#             nest_utils.assert_same_structure(
#                 experience,
#                 self.collect_data_spec,
#                 message="experience and collect_data_spec structures do not match")
#
#         if self._enable_functions:
#             preprocessed_sequence = self._preprocess_sequence_fn(experience)
#         else:
#             preprocessed_sequence = self._preprocess_sequence(experience)
#
#         if self._validate_args:
#             nest_utils.assert_same_structure(
#                 preprocessed_sequence,
#                 self.training_data_spec,
#                 message=("output of preprocess_sequence and training_data_spec "
#                          "structures do not match"))
#
#         return preprocessed_sequence
#
#     def __getattr__(self, name):
#         return getattr(self._agent, name)



# class SingleAgent:#TODO Decide what to do(maybe make it as subclass of tfagent
#
#     _precision = 10
#     _length = _precision * 2 + 1
#
#     def __init__(self,
#                  tf_agent: TFAgent,
#                  vehicle_distribution: VehicleDistributionList,
#                  capacity: int = 100,
#                  name: str = "DefaultAgent",
#                  coefficient_function = None,
#                  replay_buffer: ReplayBuffer = None,
#                  ):
#         self._name = name
#         # self._agent_id = int(filter(str.isdigit, name)
#         self._agent_id = int("".join(item for item in list(filter(str.isdigit, name))))
#
#         self._agent = MyAgentWrapper(tf_agent, self._agent_id)
#         self._agent.initialize()
#
#         self._agent.train = common.function(self._agent.train)
#
#         self.collect_policy = self._agent.collect_policy
#         self.eval_policy = self._agent.eval_policy
#
#         self._capacity = capacity
#         self._train_parking = Parking(self._capacity, 'train')
#         self._eval_parking = Parking(self._capacity, 'eval')
#
#         self._checkpointer = None
#
#         self._vehicle_distribution = vehicle_distribution
#         self._avg_vehicle_list = [0.0] * 24 if not vehicle_distribution else vehicle_distribution.avg_vehicles_list
#         self._coeffecient_calculator = coefficient_function if coefficient_function \
#                                                             else lambda x: math.sin(math.pi / 6 * x) / 2 + 0.5
#
#
#         self._state = {
#             "time_of_day": 0
#         }
#
#         self._replay_buffer = replay_buffer or None
#
#     @property
#     def checkpointer(self):
#         return self._checkpointer
#
#     @checkpointer.setter
#     def checkpointer(self, path):
#         self._checkpointer = MyCheckpointer(
#             ckpt_dir=path,
#             agent=self._agent,
#             policy=self.eval_policy,
#         )
#
#     @property
#     def name(self):
#         return self._name
#
#     @property
#     def replay_buffer(self):
#         return self._replay_buffer
#
#     @property
#     def agent(self):
#         return self._agent
#
#     def get_action(self, time_step: TimeStep, train_mode=True):
#         (policy, parking) = (self._agent.collect_policy, self._train_parking) if train_mode \
#                                                                               else (self._agent.policy,
#                                                                                     self._eval_parking)
#         observation = self._augment_observation(time_step, parking)
#         action_step = int(policy(observation))
#         max_coefficient, threshold_coefficient, min_coefficient = observation[:3]
#         step = (max_coefficient - min_coefficient) / (self._length - 1)
#         charging_coefficient = my_round(action_step * step + min_coefficient, 4)
#
#         is_charging = charging_coefficient > threshold_coefficient
#         is_buying = charging_coefficient > 0
#         self._update_avg_demand_list()
#         max_energy = (
#             parking.get_next_max_charge() if is_buying else parking.get_next_max_discharge()
#         )
#         new_energy = my_round(max_energy * charging_coefficient, 2)
#         self._update_parking(parking, is_charging, new_energy, charging_coefficient, threshold_coefficient)
#         return new_energy
#
#
#
#     def _update_avg_demand_list(self): #TODO implement way to calculate expected consumption at given time
#         if not self._vehicle_distribution:
#             episodes = int(self._state["global_step"] / 24)
#             self._avg_vehicles_list[self._state["time_of_day"]] = self._parking.get_current_vehicles() / \
#                                                                  (episodes + 1) + \
#                                                                  self._avg_vehicles_list[self._state["time_of_day"]] * \
#                                                                  (episodes / (episodes +1))
#         GARAGE_LIST.append(self._avg_vehicles_list[self._state["time_of_day"]])
#
#     def _update_parking(self, parking, is_charging, new_energy, charging_coefficient, threshold_coefficient):
#         available_energy = new_energy + parking.get_next_min_discharge() - parking.get_next_min_charge()
#         max_non_emergency_charge = parking.get_next_max_charge() - parking.get_next_min_charge()
#         max_non_emergency_discharge = parking.get_next_max_discharge() - parking.get_next_min_discharge()
#
#         update_coefficient = my_round(
#             available_energy / (max_non_emergency_charge if is_charging > 0 else max_non_emergency_discharge)
#             if my_round(abs(charging_coefficient - threshold_coefficient), 2) > 0.02
#             else 0,
#             2,
#         )
#         parking.update(update_coefficient)
#
#     def add_new_cars(self, train_mode=True):  #TODO decide on way to choose between distributions
#         parking = self._train_parking if train_mode else self._eval_parking
#         if self._state["time_of_day"] > 21:
#             return
#         if train_mode:
#             day_coefficient = self._coeffecient_calculator(self._state["time_of_day"])
#             new_cars = max(0, int(np.random.normal(10 * day_coefficient, 2 * day_coefficient)))
#             try:
#                 for _ in range(new_cars):
#                     v = self._create_vehicle()
#                     parking.assign_vehicle(v)
#             except ParkingIsFull:
#                 print("Parking is full no more cars added")
#
#         else:
#             if self._vehicle_distribution:
#                 try:
#                     new_cars = self._vehicle_distribution[self._state["global_step"]]
#                 except IndexError:
#                     print("No cars left in distribution. Probably environment was reset one more time than needed.")
#
#                 try:
#                     for total_stay, initial_charge, target_charge in new_cars:
#                         v = self._create_vehicle(total_stay, initial_charge, target_charge)
#                         parking.assign_vehicle(v)
#                 except ParkingIsFull:
#                     print("Parking is full no more cars added")
#                 except UnboundLocalError:
#                     print("No more cars in vehicle distribution, none added to the Parking")
#
#     def _calculate_vehicle_distribution(self, parking: Parking):
#         vehicle_departure_distribution = [0 for _ in range(12)]
#         for v in parking._vehicles:
#             vehicle_departure_distribution[v.get_time_before_departure() - 1] += 1
#         current_freq = 0
#         length = len(vehicle_departure_distribution)
#         for i, f in enumerate(reversed(vehicle_departure_distribution)):
#             current_freq += f
#             vehicle_departure_distribution[length - i - 1] = current_freq
#
#         return list(map(lambda x: x / self._capacity, vehicle_departure_distribution))
#
#
#     def _create_vehicle(self, total_stay_override=None, initial_charge=None, target_charge=None):
#         total_stay = (
#             min(24 - self._state["time_of_day"], random.randint(7, 10)) #Changed from 24 to allow cars to leave at 00:00
#             if not total_stay_override
#             else total_stay_override
#         )
#         min_charge = 0
#         max_charge = 60
#         initial_charge = initial_charge or my_round(6 + random.random() * 20, 2)
#         target_charge = target_charge or my_round(34 + random.random() * 20, 2)
#
#         return Vehicle(initial_charge, target_charge, total_stay, max_charge, min_charge)
#
#     def _augment_observation(self, time_step, parking):
#         next_max_charge = parking.get_next_max_charge()
#         next_min_charge = parking.get_next_min_charge()
#         next_max_discharge = parking.get_next_max_discharge()
#         next_min_discharge = parking.get_next_min_discharge()
#         max_charging_rate = parking.get_max_charging_rate()
#         max_discharging_rate = parking.get_max_discharging_rate()
#         max_acceptable = next_max_charge - next_min_discharge
#         min_acceptable = next_max_discharge - next_min_charge
#
#         max_acceptable_coefficient = (
#             max_acceptable
#             / (next_max_charge if max_acceptable > 0 else next_max_discharge)
#             if max_acceptable != 0
#             else 0
#         )
#         min_acceptable_coefficient = (
#             min_acceptable
#             / (next_max_charge() if min_acceptable < 0 else next_max_discharge())
#             if min_acceptable != 0
#             else 0
#         )
#         temp_diff = next_min_charge() - next_min_discharge()
#         threshold_coefficient = (
#             temp_diff
#             / (next_max_charge() if temp_diff > 0 else next_max_discharge())
#             if temp_diff != 0
#             else 0
#         )
#
#         return np.array(
#             [
#                 max_acceptable_coefficient,
#                 threshold_coefficient,
#                 -min_acceptable_coefficient,
#                 *time_step[:12],
#                 *self._calculate_vehicle_distribution(parking),
#                 next_max_charge / max_charging_rate / self._capacity,
#                 next_min_charge / max_charging_rate / self._capacity,
#                 next_max_discharge / max_discharging_rate/ self._capacity,
#                 next_min_discharge / max_discharging_rate/ self._capacity,
#                 parking.get_charge_mean_priority(),
#                 parking.get_discharge_mean_priority(),
#                 time_step[12],
#             ]
#         )

def create_single_agent(cls: type,
                        vehicle_distribution: List,
                        capacity: int,
                        name: str,
                        num_of_agents: int,
                        checkpoint_dir: str,
                        coefficient_function: Callable = lambda x: math.sin(math.pi / 6 * x) / 2 + 0.5,
                        *args,
                        **kwargs,):

    class_list = [cls]
    temp = cls
    while temp.__bases__:
        class_list.append(temp.__bases__[0])
        temp = temp.__bases__[0]

    if not any([i == TFAgent for i in class_list]):
        raise Exception('Provided Class is not TFAgent.'
                        'Class provided: {}'.format(cls.__class__.__name__))

    class SingleAgent(cls):
        def __init__(self,
                     vehicle_distribution: List,
                     coefficient_function: Callable,
                     checkpoint_dir: str,
                     capacity: int = 100,
                     name: str = "DefaultAgent",
                     *args,
                     **kwargs):

            self._name = name
            self._agent_id = int("".join(item for item in list(filter(str.isdigit, name))))
            if not self._agent_id:
                raise Exception('Agent name must have an integer'
                                ' to denote the agents unique id.'
                                'For example: Agent_1')

            #Parking fields. Can be removed in a different setting
            self._capacity = capacity
            self._train_parking = Parking(self._capacity, 'train')
            self._eval_parking = Parking(self._capacity, 'eval')
            self._train_vehicle_generator = vehicle_arrival_generator(coefficient_function=coefficient_function,
                                                                      vehicle_list=None)
            self._eval_vehicle_generator = vehicle_arrival_generator(coefficient_function=None,
                                                                     vehicle_list=vehicle_distribution)
            self._private_observations = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            self._private_actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)

            self.checkpointer = Checkpointer(
                ckpt_dir=checkpoint_dir + self._name,
                max_to_keep=1,
                agent=self,
            ) #TODO decide on weather it's needed
            #TODO remove avg vehicle list, there should not be access to global variables

            self._time_of_day = tf.constant(0, dtype=tf.int32)

            super(cls, self).__init__(*args, **kwargs)
            self.policy.action = self._action_wrapper(self.policy.action, False)
            self.collect_policy.action = self._action_wrapper(self.collect_policy.action, True)

            buffer_time_step_spec = TimeStep(
                    step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
                    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
                    reward=tensor_spec.TensorSpec(shape=(num_of_agents,), dtype=np.float32),
                    observation=tensor_spec.BoundedTensorSpec(shape=(13,), dtype=np.float32, minimum=-1., maximum=1.),
                )

            # buffer_action_spec = tensor_spec.TensorSpec(shape=(num_of_agents,),
            #                                             dtype=tf.float32)
            buffer_info_spec = tensor_spec.TensorSpec(shape=(),
                                                      dtype=tf.int32)
            self._collect_data_context = data_converter.DataContext(
                time_step_spec=buffer_time_step_spec,
                action_spec=(),
                info_spec=buffer_info_spec,
            )

        def _initialize(self):
            self._add_new_cars(train_mode=True)
            self._add_new_cars(train_mode=False)
            super()._initialize()

        def _add_new_cars(self, train_mode=True):  # TODO decide on way to choose between distributions
            if self._time_of_day >= 24:
                return
            parking, generator = (self._train_parking,
                                  self._train_vehicle_generator) if train_mode else (self._eval_parking,
                                                                                     self._eval_vehicle_generator)
            new_cars, _ = next(generator)
            try:
                for total_stay, initial_charge, target_charge in new_cars:
                    # v = Vehicle(initial_charge, target_charge, total_stay, 60.0, 0.0)
                    v = VehicleFields(initial_charge, target_charge, total_stay, 60.0, 0.0)
                    parking.assign_vehicle(v)
            except ParkingIsFull:
                print("Parking is full no more cars added")

        def _train(self, experience: types.NestedTensor,
                   weights: types.Tensor) -> LossInfo:
            return super()._train(self.preprocess_sequence(experience), weights)

        def _get_load(self, action_step: tf.Tensor, observation: tf.Tensor, collect_mode = True):
            parking = self._train_parking if collect_mode else self._eval_parking
            length = tf.constant((21.0), dtype=tf.float32)
            # max_coefficient, threshold_coefficient, min_coefficient  = observation[13:16]
            max_coefficient = tf.convert_to_tensor(observation[13], dtype=observation.dtype)
            threshold_coefficient = tf.convert_to_tensor(observation[14], dtype=observation.dtype)
            min_coefficient =tf.convert_to_tensor(observation[15], dtype=observation.dtype)
            step = (max_coefficient - min_coefficient) / (length-1.0)
            charging_coefficient = tf.cast(action_step, dtype=tf.float32) * step + min_coefficient

            charging_coefficient = my_round(charging_coefficient, 4)

            is_charging = tf.less_equal(threshold_coefficient, charging_coefficient)
            is_buying = tf.less_equal(tf.constant(0.0), charging_coefficient)
            max_energy = parking.get_next_max_charge() if is_buying else parking.get_next_max_discharge()
            new_energy = my_round(max_energy * charging_coefficient, 2)

            self._update_parking(parking,
                                 tf.squeeze(is_charging),
                                 tf.squeeze(new_energy),
                                 tf.squeeze(charging_coefficient),
                                 tf.squeeze(threshold_coefficient))
            return new_energy

        def _action_wrapper(self, action, collect=True):
            """
            Wraps the action method of a policy to allow it to consume
            timesteps from the single buffer, augmenting the observation
            with the saved garage state
            """
            def wrapped_action(time_step: TimeStep,
                               policy_state: types.NestedTensor = (),
                               seed: Optional[types.Seed] = None,) -> PolicyStep:
                train_mode = True
                try:
                    assert_matching_dtypes_and_inner_shapes(time_step,
                                                            self._time_step_spec,
                                                            allow_extra_fields=True,
                                                            caller=self,
                                                            tensors_name="`experience`",
                                                            specs_name="`train_argspec`")
                except ValueError:
                    parking_obs = tf.convert_to_tensor(
                        self._get_parking_observation(self._train_parking if collect
                                                                          else self._eval_parking),
                        dtype=time_step.observation.dtype,
                    )
                    augmented_obs = tf.concat(
                        (tf.squeeze(time_step.observation),
                        parking_obs),
                        0,
                    )
                    new_time_step = time_step._replace(observation=augmented_obs)
                    new_time_step = TimeStep(
                        **{k: tf.expand_dims(tf.squeeze(v), 0) for k, v in new_time_step._asdict().items()})

                    step = action(
                        new_time_step,
                        policy_state,
                        seed,)

                    load = self._get_load(step.action, tf.squeeze(augmented_obs), collect)
                    self._time_of_day = tf.cond(time_step.is_last(),
                                                lambda: 0,
                                                lambda: self._time_of_day + 1)
                    self._add_new_cars()
                    self._private_observations = tf.cond(time_step.is_last(),
                                                         lambda: self._private_observations,
                                                         lambda: self._private_observations.write(
                                                             self._private_observations.size(),
                                                             parking_obs))
                    self._private_actions = tf.cond(time_step.is_last(),
                                                    lambda: self._private_actions,
                                                    lambda: self._private_actions.write(
                                                        self._private_actions.size(),
                                                        tf.squeeze(step.action)))
                    return step.replace(action=tf.cond(tf.equal(tf.rank(tf.squeeze(load)), 0),
                                                       lambda: tf.expand_dims(tf.squeeze(load), 0),
                                                       lambda: tf.squeeze(load)))
                else:
                    return action(time_step, policy_state, seed)

            return wrapped_action

        def _preprocess_sequence(self, experience: trajectory.Trajectory):
            """
            Trajectories from the buffer contain just the market environment data
            (prices), also the reward and action of every agent. Augments observation
            with parking state, also, trims the unnecessary rewards and actions and
            removes the policy info (which is used to fetch the parking state from
            the agents field _private_observations)
            """
            parking_obs = tf.gather(self._private_observations.stack(), experience.policy_info)
            actions = tf.gather(self._private_actions.stack(), experience.policy_info)
            augmented_obs = tf.concat(
                (experience.observation,
                 parking_obs),
                axis=tf.rank(parking_obs) - 1
            )
            agent_reward = experience.reward[..., self._agent_id - 1]
            # agent_action = experience.action[..., self._agent_id - 1:self._agent_id]
            return experience.replace(observation=augmented_obs,
                                      policy_info=(),
                                      reward=agent_reward,
                                      action=actions,)

        def _calculate_vehicle_distribution(self, parking: Parking):
            vehicle_departure_distribution = [0 for _ in range(12)]
            for v in parking._vehicles:
                v = VehicleFields(*tf.unstack(v))
                vehicle_departure_distribution[tf.cast(v.time_before_departure - 1, tf.int32)] += 1
            current_freq = 0
            length = len(vehicle_departure_distribution)
            for i, f in enumerate(reversed(vehicle_departure_distribution)):
                current_freq += f
                vehicle_departure_distribution[length - i - 1] = current_freq

            return list(map(lambda x: x / self._capacity, vehicle_departure_distribution))

        def _get_parking_observation(self, parking: Parking):
            next_max_charge = parking.get_next_max_charge()
            next_min_charge = parking.get_next_min_charge()
            next_max_discharge = parking.get_next_max_discharge()
            next_min_discharge = parking.get_next_min_discharge()
            max_charging_rate = parking.get_max_charging_rate()
            max_discharging_rate = parking.get_max_discharging_rate()
            max_acceptable = next_max_charge - next_min_discharge
            min_acceptable = next_max_discharge - next_min_charge

            # max_acceptable_coefficient = (
            #     max_acceptable
            #     / (next_max_charge if max_acceptable > 0 else next_max_discharge)
            #     if max_acceptable != 0
            #     else 0
            # )
            max_acceptable_coefficient = tf.cond(tf.not_equal(max_acceptable, 0.0),
                                                 lambda: max_acceptable / tf.cond(tf.less(0.0, max_acceptable),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)
            # min_acceptable_coefficient = (
            #     min_acceptable
            #     / (next_max_charge if min_acceptable < 0 else next_max_discharge)
            #     if min_acceptable != 0
            #     else 0
            # )
            min_acceptable_coefficient = tf.cond(tf.not_equal(min_acceptable, 0.0),
                                                 lambda: min_acceptable / tf.cond(tf.less(min_acceptable, 0.0),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)

            temp_diff = next_min_charge - next_min_discharge
            # threshold_coefficient = (
            #     temp_diff
            #     / (next_max_charge if temp_diff > 0 else next_max_discharge)
            #     if temp_diff != 0
            #     else 0
            # )
            threshold_coefficient = tf.cond(tf.not_equal(temp_diff, 0.0),
                                                 lambda: temp_diff / tf.cond(tf.less(0.0, temp_diff),
                                                                                  lambda: next_max_charge,
                                                                                  lambda: next_max_discharge),
                                                 lambda: 0.0)

            return tf.stack([
                 tf.convert_to_tensor(i, dtype=tf.float32) for i in [
                    max_acceptable_coefficient,
                    threshold_coefficient,
                    -min_acceptable_coefficient,
                    *self._calculate_vehicle_distribution(parking),
                    next_max_charge / max_charging_rate / self._capacity,
                    next_min_charge / max_charging_rate / self._capacity,
                    next_max_discharge / max_discharging_rate / self._capacity,
                    next_min_discharge / max_discharging_rate / self._capacity,
                    parking.get_charge_mean_priority(),
                    parking.get_discharge_mean_priority(),
                ]
            ],
            axis = 0)

        def _update_parking(self, parking: Parking,
                            is_charging,
                            new_energy,
                            charging_coefficient,
                            threshold_coefficient):
            available_energy = new_energy + parking.get_next_min_discharge() - parking.get_next_min_charge()
            max_non_emergency_charge = parking.get_next_max_charge() - parking.get_next_min_charge()
            max_non_emergency_discharge = parking.get_next_max_discharge() - parking.get_next_min_discharge()
            update_coefficient = tf.cond(
                tf.less(0.02, my_round(tf.math.abs(charging_coefficient - threshold_coefficient), 2)),
                lambda: available_energy / tf.cond(is_charging,
                                                   lambda: max_non_emergency_charge,
                                                   lambda: max_non_emergency_discharge),
                lambda: tf.constant(0.0)
            )
            update_coefficient = my_round(update_coefficient, 2)
            parking.update(update_coefficient)

    return SingleAgent(
        vehicle_distribution=list(vehicle_distribution),
        coefficient_function=coefficient_function,
        capacity=capacity,
        name=name,
        checkpoint_dir=checkpoint_dir,
        *args,
        **kwargs,
    )
