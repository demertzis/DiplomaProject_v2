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
# market_env_train.reset()
# market_env_eval.reset()

vehicles = create_vehicle_distribution()#TODO make vehicles a new class inherited from list with method to return avg

charge_list: List[np.float32] = []
agent_list: List[DQNPolicy] = []

# garage_env_list = List[V2GEnvironment] = []
for _ in range(NUMBER_OF_AGENTS):
    next_agent = agent_list[0] if len(agent_list) > 0 else None
    train_env = V2GEnvironment(capacity=100, mode="train", name="Train_Env_" + str(NUMBER_OF_AGENTS -_), power_market_env=market_env_train, next_agent=next_agent,
                               charge_list=charge_list)
    eval_env = V2GEnvironment(capacity=200, mode='eval', name="Eval_Env_" + str(NUMBER_OF_AGENTS - _), vehicle_distribution=vehicles, power_market_env=market_env_eval,
                              next_agent=next_agent, charge_list=charge_list)

    new_agent = DQNPolicy(train_env, eval_env, name="Agent_" + str(NUMBER_OF_AGENTS - _), charge_list=charge_list, model_dir='pretrained_networks/model.keras')

    agent_list.insert(0, new_agent)

# market_env_train.change_reward_function(1)
# market_env_eval.change_reward_function(1)


agent_list[0].train()