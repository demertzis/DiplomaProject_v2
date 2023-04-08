import json

import tensorflow as tf
# from tensorflow.python.keras.optimizers import
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories.time_step import TimeStep

from app.models.energy import EnergyCurve
from app.models.pwr_market_env import PowerMarketEnv
from app.policies.multiple_agents import MultipleAgents
from app.policies.reward_functions import *
from app.utils import VehicleDistributionList, create_dqn_agent
from config import NUMBER_OF_AGENTS

energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')


train_env = PowerMarketEnv(energy_curve_train, vanilla, NUMBER_OF_AGENTS, True)
eval_env = PowerMarketEnv(energy_curve_eval, vanilla, NUMBER_OF_AGENTS, False)


with open('data/vehicles_old.json') as file:
    vehicles = VehicleDistributionList(json.load(file))

learning_rate = 1e-3

single_agent_time_step_spec = tensor_spec.from_spec(TimeStep(
    step_type=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
    discount=array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
    reward=array_spec.ArraySpec(shape=(), dtype=np.float32),
    observation=array_spec.BoundedArraySpec(shape=(34,), dtype=np.float32, minimum=-1., maximum=1.),
))

single_agent_action_spec = tensor_spec.from_spec(array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=20, name="action"
        ))

kwargs = {
    'time_step_spec': single_agent_time_step_spec,
    'action_spec': single_agent_action_spec,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
    'td_errors_loss_fn': common.element_wise_squared_loss,
    'target_update_tau': 0.001,
    'target_update_period': 1,
}

agent_list = [create_dqn_agent('pretrained_networks/model.keras', **kwargs) for _ in range(NUMBER_OF_AGENTS)]

multi_agent = MultipleAgents(train_env, eval_env, agent_list)

multi_agent.train()
