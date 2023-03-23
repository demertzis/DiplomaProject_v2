import json

from app.models.energy import EnergyCurve
from app.utils import VehicleDistributionList
from typing import List
from app.policies.smart_charger import DummyV2G
from app.policies.dummy_v2g import SmartCharger
from app.policies.utils import metrics_visualization
from app.policies.dqn import DQNPolicy
from app.models.garage_env import V2GEnvironment
from config import NUMBER_OF_AGENTS
import numpy as np
from gym.envs.registration import register
import gym

energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')

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

train_env = V2GEnvironment(300, './data/GR-data-new.csv', 'train')

with open("data/vehicles.json", "rb") as file:
    vehicles = VehicleDistributionList(json.load(file))


charge_list: List[np.float32] = []
agent_list: List[DQNPolicy] = []

# DQN
eval_env = V2GEnvironment(300, './data/GR-data-new.csv', 'eval', vehicles)

# garage_env_list = List[V2GEnvironment] = []
for _ in range(NUMBER_OF_AGENTS):
    next_agent = agent_list[0] if len(agent_list) > 0 else None
    train_env = V2GEnvironment(capacity=100, mode="train", name="Train_Env_" + str(NUMBER_OF_AGENTS -_), power_market_env=market_env_train, next_agent=next_agent,
                               charge_list=charge_list)
    eval_env = V2GEnvironment(capacity=200, mode='eval', name="Eval_Env_" + str(NUMBER_OF_AGENTS - _), vehicle_distribution=vehicles, power_market_env=market_env_eval,
                              next_agent=next_agent, charge_list=charge_list)

    new_agent = DQNPolicy(train_env, eval_env, name="Agent_" + str(NUMBER_OF_AGENTS - _), charge_list=charge_list)

    agent_list.insert(0, new_agent)


# agent_list[0].trickle_down('raw_eval_env.hard_reset()')
# agent_list[0].trickle_down('eval_env.reset()')
# for agent in agent_list: agent.set_policy()

# agent_list[0].compute_eval_list(10)
# ret = agent_list[0].compute_avg_agent_loss()
# print(f"Dqn agents: The average return: {ret}")
print(f"Dqn agents: The average return: {agent_list[0].evaluate_policy()}")
# metrics_visualization(eval_env.get_metrics(), 0, 'dqn')

# Dummy V2G
# agent_list[0].trickle_down('raw_eval_env.hard_reset()')
# agent_list[0].trickle_down('eval_env.reset()')
# for agent in agent_list: agent.set_policy(DummyV2G(0.5))
#
# agent_list[0].compute_eval_list(10)
# ret = agent_list[0].compute_avg_agent_loss()
# print(f"Dummy agents: The average return: {ret}")
print(f"Dummy agents: The average return: {agent_list[0].evaluate_policy(DummyV2G(0.5))}")
# metrics_visualization(eval_env.get_metrics(), 0, 'dummy_v2g')

# Smart Charger
# agent_list[0].trickle_down('raw_eval_env.hard_reset()')
# agent_list[0].trickle_down('eval_env.reset()')
# for agent in agent_list: agent.set_policy(SmartCharger(0.5))
#
# agent_list[0].compute_eval_list(10)
# ret = agent_list[0].compute_avg_agent_loss()
# print(f"Smart_agents: The average return: {ret}")
print(f"Smart_agents: The average return: {agent_list[0].evaluate_policy(SmartCharger(0.5))}")
# metrics_visualization(eval_env.get_metrics(), 0, 'smart_charger')
