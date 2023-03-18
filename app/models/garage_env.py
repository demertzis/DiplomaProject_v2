# from app.policies.utils import sigmoid
import json
import math
import random
from typing import Any, Dict, List

import numpy as np
# import statistics
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from config import GARAGE_LIST, BATTERY_CAPACITY, AVG_CHARGING_RATE, DISCOUNT

from app.error_handling import ParkingIsFull
from app.models.parking import Parking
from app.models.power_market_env import PowerMarketEnv
# from app.policies.dqn import DQNPolicy
from app.models.vehicle import Vehicle

from app.utils import VehicleDistributionList


class V2GEnvironment(PyEnvironment):
    _precision = 10
    _length = _precision * 2 + 1
    _battery_cost = 120
    _battery_capacity = BATTERY_CAPACITY

    def __init__(self,
                 capacity: int,
                 mode: str,
                 name: str = "Default_Garage",
                 vehicle_distribution: VehicleDistributionList = None,
                 # energy_curve: EnergyCurve = None,
                 power_market_env: PowerMarketEnv = None,
                 next_agent: Any = None,
                 charge_list: List[np.float32] = [],
                 coefficient_function = None):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._length - 1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(34,), dtype=np.float32, minimum=-1., maximum=1., name="observation"
        )
        self._time_step_spec = time_step.TimeStep(
            step_type=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
            discount=array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
            reward=array_spec.ArraySpec(shape=(), dtype=np.float32),
            observation=self._observation_spec,
        )
        self._power_market_env = power_market_env
        self._mode = mode
        self._name = name

        self._state = {
            "time_of_day": 0,
            "step": 0,
            "global_step": 0,
            # "metrics": {
            #     "loss": [],
            #     "num_of_vehicles": [],
            #     "cost": [],
            #     "overcharged_time_per_car": [],
            #     # "c_rate_per_car": [],
            #     # "cycle_degradation": [],
            #     # "age_degradation": [],
            #     "actions": [],
            #     "charging_coefficient": [],
            #     "test_sum": 0,
            # },
        }
        self._capacity = capacity
        self._parking = Parking(capacity, name)
        self._avg_vehicles_list = [0.0] * 24 if not vehicle_distribution else vehicle_distribution.avg_vehicles_list
        # self._energy_curve = energy_curve
        self._coeffecient_calculator = (lambda x: math.sin(math.pi / 6 * x) / 2 + 0.5) if coefficient_function == None \
                                                                                     else coefficient_function
        self._vehicle_distribution = vehicle_distribution
        self._next_agent: Any = next_agent
        self._charge_list: List[np.float32] = charge_list

    # def get_metrics(self):
    #     return self._state["metrics"]

    def _update_avg_demand_list(self): #TODO implement way to calculate expected consumption at given time
        if not self._vehicle_distribution:
            episodes = int(self._state["global_step"] / 24)
            self._avg_vehicles_list[self._state["time_of_day"]] = self._parking.get_current_vehicles() / \
                                                                 (episodes + 1) + \
                                                                 self._avg_vehicles_list[self._state["time_of_day"]] * \
                                                                 (episodes / (episodes +1))
        GARAGE_LIST.append(self._avg_vehicles_list[self._state["time_of_day"]])

    def next_agent(self):
        return self._next_agent

    # def reset_metrics(self):
    #     self._state["metrics"] = {
    #         "loss": [],
    #         "num_of_vehicles": [],
    #         "cost": [],
    #         "overcharged_time_per_car": [],
    #         # "c_rate_per_car": [],
    #         # "cycle_degradation": [],
    #         # "age_degradation": [],
    #         "actions": [],
    #         "charging_coefficient": [],
    #         "test_sum": 0,
    #     }

    def time_step_spec(self):
        return self._time_step_spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _normalize_array(self, energy_list):
        max_value = max(energy_list)
        min_value = min(energy_list)
        diff = max_value - min_value
        if diff == 0:
            print('')
        return list(map(lambda x: np.float32((x - min_value) / diff) if diff != 0 else np.float32(0.0), energy_list))

    def hard_reset(self):
        self._state["global_step"] = 0
        self._power_market_env.hard_reset()
        # self._energy_curve.reset()
        # self.reset_metrics()
        self._avg_vehicles_list = [0.0] * 24 if not self._vehicle_distribution else self._avg_vehicles_list


    def _reset(self) -> time_step.TimeStep:
        # print("hi " + self._name)
        self._state["time_of_day"] = 0
        self._state["step"] = 0

        #added cars from the first hour
        self._add_new_cars()

        observation = self._power_market_env.reset()
        energy_costs = observation[:24]
        energy_cost_intra_day = observation[24]

        normalized_energy = self._normalize_array(list(energy_costs[:12]) + [energy_cost_intra_day])

        # energy_costs = self._energy_curve.get_next_batch()
        # energy_cost_intra_day = self._energy_curve.get_current_cost_intra_day()

        return time_step.TimeStep(
            step_type=time_step.StepType.FIRST,
            reward=np.array(0.0, dtype=np.float32),
            discount=np.array(0, dtype=np.float32),
            observation=np.array(
                [
                    1,
                    0,
                    -1,
                    *normalized_energy[:12],
                    *self._calculate_vehicle_distribution(),
                    self._parking.get_next_max_charge() / self._parking.get_max_charging_rate() / self._capacity,
                    self._parking.get_next_min_charge() / self._parking.get_max_charging_rate() / self._capacity,
                    self._parking.get_next_max_discharge() / self._parking.get_max_discharging_rate() / self._capacity,
                    self._parking.get_next_min_discharge() / self._parking.get_max_discharging_rate() / self._capacity,
                    self._parking.get_charge_mean_priority(),
                    self._parking.get_discharge_mean_priority(),
                    normalized_energy[12],
                ],
                dtype=np.float32,
            ),
        )

    def _step(self, action: int) -> time_step.TimeStep:
        idx = int(action)
        max_coefficient, threshold_coefficient, min_coefficient = self.current_time_step().observation[:3]
        step = (max_coefficient - min_coefficient) / (self._length - 1)
        # print(max_coefficient, threshold_coefficient, min_coefficient, step)
        charging_coefficient = round(idx * step + min_coefficient, 4)

        # if charging_coefficient < 0:
        #     # print("")
        #     ...

        is_charging = charging_coefficient > threshold_coefficient
        is_buying = charging_coefficient > 0

        num_of_vehicles = len(self._parking._vehicles)

        # self._state["metrics"]["cost"].append(0)
        # # self._state["metrics"]["cycle_degradation"].append(0)
        # # self._state["metrics"]["age_degradation"].append(0)
        # self._state["metrics"]["num_of_vehicles"].append(num_of_vehicles)
        # self._state["metrics"]["actions"].append(idx)
        # self._state["metrics"]["charging_coefficient"].append(charging_coefficient)
        
        self._update_avg_demand_list()

        if num_of_vehicles != 0:
            try:
                # current_cost = self._energy_curve.get_current_cost() / 1000

                max_energy = (
                    self._parking.get_next_max_charge() if is_buying else self._parking.get_next_max_discharge()
                )

                new_energy = round(max_energy * charging_coefficient, 2)

                self._charge_list.insert(0, np.float32(new_energy))
                if self._next_agent is not None:
                    # if self._mode == 'eval':
                    # self._next_agent.eval_env.step()

                    self._next_agent.collect_step() if self._mode == 'train' else self._next_agent.collect_step(False)
                    # self._next_agent._train()
                    # self._next_agent.train_env.update_charge_list(self._charge_list.insert(0, np.float32(new_energy)))
                    # self._next_agent.eval_env.update_charge_list(self._charge_list.insert(0, np.float32(new_energy)))

                observation, current_cost, done, info = self._power_market_env.step(self._charge_list)

                # print(f"New energy: {new_energy}")

                threshold_energy = self._parking.get_next_min_discharge() - self._parking.get_next_min_charge()
                available_energy = new_energy + threshold_energy

                # print(f"Available energy: {available_energy}")
                max_non_emergency_charge = self._parking.get_next_max_charge() - self._parking.get_next_min_charge()
                max_non_emergency_discharge = (
                        self._parking.get_next_max_discharge() - self._parking.get_next_min_discharge()
                )
                # print(f"Max non emergency charge: {max_non_emergency_charge}")
                # print(f"Max non emergency discharge: {max_non_emergency_discharge}")

                update_coefficient = round(
                    available_energy / (max_non_emergency_charge if is_charging > 0 else max_non_emergency_discharge)
                    if round(abs(charging_coefficient - threshold_coefficient), 2) > 0.02
                    else 0,
                    2,
                )
                # print(
                #     f"Charg Coeff: {charging_coefficient}, New E: {new_energy}, ",
                #     f"Max energy: {max_energy}, Update coeff: {update_coefficient}",
                #     f"Available Energy: {available_energy}"
                # )
                # print(f"Update coefficient: {update_coefficient}")
                # avg_charge_levels, overcharged_time = self._parking.update(update_coefficient)
                self._parking.update(update_coefficient)

                # print(f"Available charge levels: {avg_charge_levels}")
                # print(f"Degrade rates: {degrade_rates}")

                # cost = np.float32(current_cost)
                # reward = -np.float32(current_cost)
                # cost = int(new_energy * current_cost * 100)
                # print(
                #     f"Charg Coeff: {charging_coefficient}, New E: {new_energy}, Current C: {current_cost},"
                #     + f" Total: {cost}, Vehicles: {num_of_vehicles}"
                #     + f", Avg Ch L: {sum(avg_charge_levels) / len(avg_charge_levels)}"
                # )
                # self._state["metrics"]["test_sum"] += new_energy

                # cycle_degradation_cost = 0.0
                # for degrade_rate in degrade_rates:
                #     cycle_degradation_cost += (
                #         (
                #                 self.cycle_degradation(degrade_rate / self._battery_capacity)
                #                 * self._battery_capacity
                #                 * self._battery_cost
                #         )
                #         if degrade_rate > 0
                #         else 0
                #     )
                # cycle_degradation_cost = int(cycle_degradation_cost * 100)
                #
                # age_degradation_cost = 0.0
                # for charge_level in avg_charge_levels:
                #     age_degradation_cost += (
                #             self.age_degradation(charge_level / self._battery_capacity)
                #             * self._battery_capacity
                #             * self._battery_cost
                #     )
                # age_degradation_cost = int(age_degradation_cost * 100)

                # reward = -cost
                # reward = -cost - cycle_degradation_cost - age_degradation_cost
                # reward = -cost - unmet_demand - cycle_degradation_cost - age_degradation_cost
                # reward = sigmoid( )

                # Update metrics
                # self._state["metrics"]["cost"][-1] = cost
                # # self._state["metrics"]["cycle_degradation"][-1] = cycle_degradation_cost
                # # self._state["metrics"]["age_degradation"][-1] = age_degradation_cost
                # self._state["metrics"]["overcharged_time_per_car"] += overcharged_time
                # # self._state["metrics"]["c_rate_per_car"] += [
                # #     round(d / self._battery_capacity, 4) for d in degrade_rates
                # # ]

                # print(f"Energy cost: {cost}")
                # print(f"Cycle degradation cost: {cycle_degradation_cost}")
                # print(f"Age degradation cost: {age_degradation_cost}")

            except ValueError as e:
                # print(cost, cycle_degradation_cost, age_degradation_cost, avg_charge_levels)
                print(self)
                raise e

        elif num_of_vehicles == 0:  # TODO Deal with 0 vehicles situation
            self._charge_list.insert(0, np.float32(0.0))
            if self._next_agent is not None:
                self._next_agent.collect_step() if self._mode == 'train' else self._next_agent.collect_step(False)
                # self._next_agent.train_env.update_charge_list(self._charge_list.insert(0, np.float32(new_energy)))
                # self._next_agent.eval_env.update_charge_list(self._charge_list.insert(0, np.float32(new_energy)))

            observation, current_cost, done, info = self._power_market_env.step(self._charge_list)

        # reward = -np.float32(current_cost)

        self._state["time_of_day"] += 1
        self._state["time_of_day"] %= 24
        self._state["step"] += 1
        self._state["global_step"] += 1

        if self._state["time_of_day"] != 0:
            self._add_new_cars()

        # energy_costs = np.ndarray(observation[self._state["time_of_day"]:24], dtype = np.float32)
        energy_costs = np.append(observation[self._state["time_of_day"]:24],
                                 np.full(max(0, (self._state["time_of_day"] - 12)),
                                         -1.0, dtype=np.float32))

        # energy_costs.extend(np.full((12 - len(energy_costs),), 0., dtype=np.float32))
        # if self._state["step"] == 24:
        #     energy_costs = self._energy_curve.get_current_batch()
        # else:
        #     energy_costs = self._energy_curve.get_next_batch()
        energy_cost_intra_day = observation[24]

        # energy_cost_intra_day = self._energy_curve.get_current_cost_intra_day()

        max_acceptable = self._parking.get_next_max_charge() - self._parking.get_next_min_discharge()
        min_acceptable = self._parking.get_next_max_discharge() - self._parking.get_next_min_charge()
        max_acceptable_coefficient = (
            max_acceptable
            / (self._parking.get_next_max_charge() if max_acceptable > 0 else self._parking.get_next_max_discharge())
            if max_acceptable != 0
            else 0
        )

        min_acceptable_coefficient = (
            min_acceptable
            / (self._parking.get_next_max_charge() if min_acceptable < 0 else self._parking.get_next_max_discharge())
            if min_acceptable != 0
            else 0
        )

        temp_diff = self._parking.get_next_min_charge() - self._parking.get_next_min_discharge()
        threshold_coefficient = (
            temp_diff
            / (self._parking.get_next_max_charge() if temp_diff > 0 else self._parking.get_next_max_discharge())
            if temp_diff != 0
            else 0
        )

        valid_hours = min(12, 24 - self._state["time_of_day"])
        normalized_energy = self._normalize_array(list(energy_costs[:valid_hours]) + [energy_cost_intra_day])
        normalized_energy[valid_hours:valid_hours] = [np.float32(-1.0)] * max(0, (self._state["time_of_day"] - 12))
        if done and self._state["step"] < 24:
            raise ValueError
        return time_step.TimeStep(
            step_type=time_step.StepType.MID if self._state["step"] < 24 else time_step.StepType.LAST,
            reward=np.array(-current_cost / 1000, dtype=np.float32),
            discount=np.array(DISCOUNT, dtype=np.float32                                                                                                                                                                                                                                                                                                                                                                                                        ),
            observation=np.array(
                [
                    max_acceptable_coefficient,
                    threshold_coefficient,
                    -min_acceptable_coefficient,
                    *normalized_energy[:12],
                    *self._calculate_vehicle_distribution(),
                    self._parking.get_next_max_charge() / self._parking.get_max_charging_rate() / self._capacity,
                    self._parking.get_next_min_charge() / self._parking.get_max_charging_rate() / self._capacity,
                    self._parking.get_next_max_discharge() / self._parking.get_max_discharging_rate() / self._capacity,
                    self._parking.get_next_min_discharge() / self._parking.get_max_discharging_rate() / self._capacity,
                    self._parking.get_charge_mean_priority(),
                    self._parking.get_discharge_mean_priority(),
                    normalized_energy[12],
                ],
                dtype=np.float32,
            ),
        )

    def _calculate_vehicle_distribution(self):
        vehicle_departure_distribution = [0 for _ in range(12)]
        for v in self._parking._vehicles:
            vehicle_departure_distribution[v.get_time_before_departure() - 1] += 1
        currentFreq = 0
        length = len(vehicle_departure_distribution)
        for i, f in enumerate(reversed(vehicle_departure_distribution)):
            if f != 0:
                currentFreq += f
            vehicle_departure_distribution[length - i - 1] = currentFreq

        return list(map(lambda x: x / self._capacity, vehicle_departure_distribution))

    def _add_new_cars(self):  #TODO decide on way to choose betwen distributions
        if self._state["time_of_day"] > 21:
            return

        if self._vehicle_distribution:
            try:
                new_cars = self._vehicle_distribution[self._state["global_step"]]
            except IndexError:
                print("No cars left in distribution. Probably environment was reset one more time than needed.")

            try:
                for total_stay, initial_charge, target_charge in new_cars:
                    v = self._create_vehicle(total_stay, initial_charge, target_charge)
                    self._parking.assign_vehicle(v)
            except ParkingIsFull:
                print("Parking is full no more cars added")
            except UnboundLocalError:
                print("No more cars in vehicle distribution, none added to the Parking")
        else:
            # day_coefficient = math.sin(math.pi / 6 * self._state["time_of_day"]) / 2 + 0.5#TODO probably add constant
            day_coefficient = self._coeffecient_calculator(self._state["time_of_day"])
            new_cars = max(0, int(np.random.normal(10 * day_coefficient, 2 * day_coefficient)))
            try:
                for _ in range(new_cars):
                    v = self._create_vehicle()
                    self._parking.assign_vehicle(v)
            except ParkingIsFull:
                print("Parking is full no more cars added")

    def _create_vehicle(self, total_stay_override=None, initial_charge=None, target_charge=None):
        total_stay = (
            min(24 - self._state["time_of_day"], random.randint(7, 10)) #Changed from 24 to allow cars to leave at 00:00
            if not total_stay_override
            else total_stay_override
        )
        min_charge = 0
        max_charge = 60
        initial_charge = initial_charge or round(6 + random.random() * 20, 2)
        target_charge = target_charge or round(34 + random.random() * 20, 2)

        return Vehicle(initial_charge, target_charge, total_stay, max_charge, min_charge)

    # def cycle_degradation(self, c_rate):
    #     return 5e-5 * c_rate + 1.8e-5
    #
    # def age_degradation(self, soc):
    #     """
    #     Returns the age degradation percentage based on the following formula
    #
    #     ``L = 0.09 * soc * 0.02 / 900 / 24``
    #
    #     ### Returns
    #         float : the total lost capacity
    #     """
    #     return (0.09 * soc + 0.02) / 900 / 24

    def get_reward_func_name(self):
        return self._power_market_env.get_reward_func_name()

    def toJson(self) -> Dict[str, Any]:
        return {"name": self._mode, "parking": self._parking.toJson()}

    def __repr__(self) -> str:
        return json.dumps(self.toJson(), indent=4)
