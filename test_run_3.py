import json

import numpy as np
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories.time_step import TimeStep
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.networks import sequential
import app.policies.tf_reward_functions as rf

from app.models.tf_energy import EnergyCurve
from app.policies.multiple_tf_agents import MultipleAgents
from config import NUMBER_OF_AGENTS

from app.models.tf_pwr_env import TFPowerMarketEnv

from app.utils import VehicleDistributionList
from app.abstract.tf_single_agent import create_single_agent
# tf.config.run_functions_eagerly(True)

with open('data/vehicles.json') as file:
    vehicles = VehicleDistributionList(json.load(file))

single_agent_time_step_spec = tensor_spec.from_spec(TimeStep(
    step_type=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2),
    discount=array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
    reward=array_spec.ArraySpec(shape=(), dtype=np.float32),
    observation=array_spec.BoundedArraySpec(shape=(34,), dtype=np.float32, minimum=-1., maximum=1.),
))

single_agent_action_spec = tensor_spec.from_spec(array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=20, name="action"
        ))

learning_rate = 1e-3



model_dir = 'pretrained_networks/model.keras'
num_actions = 21

# if not model_dir:
layers_list = \
    [
        tf.keras.layers.Dense(units=34, activation="elu"),
        tf.keras.layers.Dense(units=128, activation="elu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            num_actions,
            activation=None,
        ),
    ]
# else:
#     keras_model = tf.keras.models.load_model(model_dir)
#     layers_list = []
#     i = 0
#     while True:
#         try:
#             layers_list.append(keras_model.get_layer(index=i))
#         except IndexError:
#             print('{0}: Total number of layers in neural network: {1}'.format('Agent-1', i))
#             break
#         except ValueError:
#             print('{0}: Total number of layers in neural network: {1}'.format('Agent-1', i))
#             break
#         else:
#             i += 1

q_net = sequential.Sequential(layers_list)

kwargs = {
    'time_step_spec': single_agent_time_step_spec,
    'action_spec': single_agent_action_spec,
    'q_network': q_net,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
    'td_errors_loss_fn': common.element_wise_squared_loss,
    'target_update_tau': 0.001,
    'target_update_period': 1,
}

reward_function = rf.vanilla
reward_name = reward_function.__name__

agent_list = []
for i in range(NUMBER_OF_AGENTS):
    agent_list.append(create_single_agent(cls=DdqnAgent,
                                          checkpoint_dir='/'.join([
                                              'new_checkpoints',
                                              str(NUMBER_OF_AGENTS) +
                                              '_AGENTS',
                                              reward_name,
                                          ]),
                                          vehicle_distribution=list(vehicles),
                                          capacity_train_garage=100,
                                          capacity_eval_garage=200,
                                          name='Agent-' + str(i+1),
                                          num_of_agents=NUMBER_OF_AGENTS,
                                          **kwargs))

energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')


train_env = TFPowerMarketEnv(energy_curve_train,
                             reward_function,
                             NUMBER_OF_AGENTS,
                             True)
eval_env = TFPowerMarketEnv(energy_curve_eval,
                            reward_function,
                            NUMBER_OF_AGENTS,
                            False)

multi_agent = MultipleAgents(train_env,
                             eval_env,
                             agent_list,)
# print(multi_agent.eval_policy())

multi_agent.train()
print('')
