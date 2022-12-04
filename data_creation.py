from typing import List

import gym
import numpy as np
from gym.envs.registration import register
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories.time_step import TimeStep

from app.models.energy import EnergyCurve
from app.models.garage_env import V2GEnvironment
from app.policies.smart_charger import SmartCharger
from app.utils import create_vehicle_distribution

energy_curve_eval = EnergyCurve('data/data_sorted_by_date.csv', 'eval')

register(
    id='PowerDataCreation-v0',
    entry_point='app.models.power_market_env:PowerMarketEnv',
    kwargs={
        'energy_curve': energy_curve_eval,
        'reward_function': 0,
        'agents': 1
    }
)
market_env_eval = gym.make('PowerDataCreation-v0')
# market_env_eval.reset()

# data_creation_env = V2GEnvironment()

for i in range(10):
    vehicles = create_vehicle_distribution(energy_curve_eval.total_episodes() * 24)

    charge_list: List[np.float32] = []

    env = V2GEnvironment(capacity=200, name='eval', vehicle_distribution=vehicles, power_market_env=market_env_eval,
                         next_agent=None, charge_list=charge_list)

    # file = open("colab_data.csv", "w")
    # writer = csv.writer(file)
    env.hard_reset()

    env = tf_py_environment.TFPyEnvironment(env)

    policy = SmartCharger(0.5)
    # policy = DummyV2G(0.5)

    data = np.empty((0, 55), np.float32)

    time_step: TimeStep = env.reset()

    reward_list: List[np.float32] = []


    def create_gaussian_outputs(action_num: int, value: np.float32):
        """
        Creates an extrapolation of state action values based on the fact that according to the "smart policy" a
        particular action is chosen, which means that it should have the highest value. Extrapolates the other values
        considering a gaussian distribution which is maximized at the value of the state action chosen and has a
        standard deviation of 20 (hear the x values of the distribution are the differenta actions (0 - 21)
        """
        gaussian = lambda x: value * np.exp(-0.5 * (x - action_num) ** 2 / (2 * 20 ** 2))
        return [np.float32(gaussian(x)) for x in range(21)]


    for _ in range(energy_curve_eval.total_episodes() * 24):
        observation = time_step.observation.numpy()[0]
        action_step = policy.action(time_step)
        action_num = int(action_step.action)

        # create rough (very) estimate of value of different actions based on the fact that action_num was taken
        action_array = np.array([0] * (action_num) + [1] + [0] * (20 - action_num))
        data_line = np.array(np.append(observation, action_array), dtype=np.float32, ndmin=2)
        data = np.append(data, data_line, axis=0)
        time_step = env.step(action_step.action)
        #TODO deal with the accumulation of rewards (check if signs are correct and consider decay factor)
        reward_list.append(time_step.reward.numpy()[0])
        if time_step.is_last():
            total_reward = sum(reward_list)
            for i in range(24):
                gaussian_values = create_gaussian_outputs(np.nonzero(data[-24 + i][34:55])[-1][-1], total_reward)
                data[-24 + i][34:55] = gaussian_values
                # for j in range(21):
                #     data[-24 + i][34 + j] = gaussian_values[j]
                total_reward -= reward_list[i]
            reward_list.clear()
            time_step = env.reset()

        charge_list.clear()

    with open("colab_data.csv", "a") as file:
        np.savetxt(file, data, delimiter=',')
        # data.savetxt(file, sep=',')

    # energy_curve_2 = energy.EnergyCurve('data/GR-data-11-20.csv', 'train')
    # print(energy_curve[:5])
