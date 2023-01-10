from typing import List

import gym
import numpy as np
from gym.envs.registration import register

from app.models.energy import EnergyCurve
from app.models.garage_env import V2GEnvironment
from app.policies.dqn import DQNPolicy
from app.utils import create_vehicle_distribution
from config import NUMBER_OF_AGENTS

energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
energy_curve_eval = EnergyCurve('data/data_sorted_by_date.csv', 'eval')

register(
    id='PowerTrain-v0',
    entry_point='app.models.power_market_env:PowerMarketEnv',
    kwargs={
        'energy_curve': energy_curve_train,
        'reward_function': 0
    }
)
register(
    id='PowerEval-v0',
    entry_point='app.models.power_market_env:PowerMarketEnv',
    kwargs={
        'energy_curve': energy_curve_eval,
        'reward_function': 0
    }
)

market_env_train = gym.make('PowerTrain-v0')
market_env_eval = gym.make('PowerEval-v0')

vehicles = create_vehicle_distribution()

charge_list: List[np.float32] = []
# garage_env_list = List[V2GEnvironment] = []
train_env = V2GEnvironment(capacity=100, name="Train_Garage", mode="train", power_market_env=market_env_train, next_agent=None,
                           charge_list=charge_list)
eval_env = V2GEnvironment(capacity=200, name='Eval_Garage', mode="eval", vehicle_distribution=vehicles, power_market_env=market_env_eval,
                          next_agent=None, charge_list=charge_list)

new_agent = DQNPolicy(train_env, eval_env,  charge_list=charge_list, model_dir='pretrained_networks/model.keras')

new_agent.train()