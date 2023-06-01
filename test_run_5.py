import itertools
import json
import math
from time import time

from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.networks import sequential
import app.policies.tf_reward_functions as rf

from app.models.tf_energy_2 import EnergyCurve
from app.policies.multiple_tf_agents_3 import MultipleAgents
from app.policies.tf_dummy_v2g import DummyV2G
from app.policies.tf_smart_charger import SmartCharger
from config import NUMBER_OF_AGENTS, MAX_BUFFER_SIZE, AVG_CHARGING_RATE

from app.models.tf_pwr_env_2 import TFPowerMarketEnv

from app.utils import VehicleDistributionListConstantShape, calculate_avg_distribution_constant_shape
from app.abstract.tf_single_agent_5 import create_single_agent
from data_creation_2 import create_train_data

# tf.debugging.set_log_device_placement(True)
# tf.config.run_functions_eagerly(True)
# tf.debugging.experimental.enable_dumpa_debug_info(
#     "/tmp/tfdbg2_logdir",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)
# tf.debugging.enable_check_numerics()
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
# with open('data/vehicles.json') as file:
#     vehicles_2 = VehicleDistributionList(json.load(file))
with open('data/vehicles_constant_shape.json') as file:
    vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))

single_agent_time_step_spec = TimeStep(
    step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int64, minimum=0, maximum=2),
    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0.0, maximum=1.0),
    reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
    observation=tensor_spec.BoundedTensorSpec(shape=(34,), dtype=tf.float32, minimum=-1., maximum=1.))

num_actions = 16
single_agent_action_spec = tensor_spec.BoundedTensorSpec(
            shape=(), dtype=tf.int64, minimum=0, maximum=num_actions-1, name="action")

model_dir = 'pretrained_networks/model_output_16.keras'
#
# layers_list = \
#     [
#         # tf.keras.layers.Dense(units=34, activation="elu"),
#         tf.keras.layers.Dense(units=64, activation="elu"),
#         tf.keras.layers.Dense(units=128, activation="elu"),
#         # tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(
#             num_actions,
#         ),
#         tf.keras.layers.Activation('linear', dtype=tf.float32)
#     ]


keras_model = tf.keras.models.load_model(model_dir)
layers_list = []
i = 0
while True:
    try:
        layers_list.append(keras_model.get_layer(index=i))
    except IndexError:
        print('{0}: Total number of layers in neural network: {1}'.format('Agent-1', i))
        break
    except ValueError:
        print('{0}: Total number of layers in neural network: {1}'.format('Agent-1', i))
        break
    else:
        i += 1

temp_model = tf.keras.Sequential(layers_list)
temp_model.build(input_shape=(1,34))
temp_target_model = tf.keras.models.clone_model(temp_model)


q_net = sequential.Sequential(temp_model.layers, name='QNetwork')
target_q_net = sequential.Sequential(temp_target_model.layers, name='TargetQNetwork')
# temp = q_net.create_variables(input_tensor_spec=single_agent_time_step_spec.observation)
# target_q_net = q_net.copy(_name='Target_Q_Network')

learning_rate = 1e-4
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=9600,
#     staircase=True,
#     decay_rate=0.9)

# learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
#     initial_learning_rate=5e-3,
#     first_decay_steps=2400,
#     t_mul=2.0,
#     alpha=0.5,
#     m_mul=0.8)

kwargs = {
    'time_step_spec': single_agent_time_step_spec,
    'action_spec': single_agent_action_spec,
    'q_network': q_net,
    'target_q_network': target_q_net,
    # 'optimizer': tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate,
    #                                                                                   amsgrad=True),),
    'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate),
    # 'td_errors_loss_fn': common.element_wise_squared_loss,
    # 'epsilon_greedy': 0.1,
    'epsilon_greedy': None,
    'boltzmann_temperature': 0.01,
    'target_update_tau': 0.01,
    'target_update_period': 10,
}

reward_function = rf.vanilla
reward_name = reward_function.__name__

ckpt_dir = '/'.join(['checkpoints',
                     str(NUMBER_OF_AGENTS) +
                     '_AGENTS',
                     reward_name])

coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5
agent_list = []
for i in range(NUMBER_OF_AGENTS):
    agent_list.append(create_single_agent(cls=DdqnAgent,
    # agent_list.append(create_single_agent(cls=MyDdqnAgent,
                                          ckpt_dir=ckpt_dir,
                                          vehicle_distribution=list(vehicles),
                                          buffer_max_size=MAX_BUFFER_SIZE,
                                          num_of_actions=num_actions,
                                          capacity_train_garage=100,
                                          capacity_eval_garage=200,
                                          name='Agent-' + str(i+1),
                                          num_of_agents=NUMBER_OF_AGENTS,
                                          coefficient_function=coefficient_function,
                                          **kwargs))

collect_avg_vehicles_list = tf.constant([0.0]*24)
days = 500
for agent in agent_list:
    collect_avg_vehicles_list += calculate_avg_distribution_constant_shape(days,
                                                                           agent.train_vehicles_generator)
collect_avg_vehicles_list = collect_avg_vehicles_list.numpy()

energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')

# collect_avg_vehicles_list = tf.constant([10.0] * 24)
train_env = TFPowerMarketEnv(energy_curve_train,
                             reward_function,
                             NUMBER_OF_AGENTS,
                             [AVG_CHARGING_RATE * v for v in collect_avg_vehicles_list],
                             True)
eval_env = TFPowerMarketEnv(energy_curve_eval,
                            reward_function,
                            NUMBER_OF_AGENTS,
                            [AVG_CHARGING_RATE * v * len(agent_list) for v in vehicles.avg_vehicles_list],
                            False)

multi_agent = MultipleAgents(train_env,
                             eval_env,
                             agent_list,
                             ckpt_dir,)
                             # SmartCharger(0.5, num_actions, single_agent_time_step_spec))

variable_list = list(itertools.chain.from_iterable([module.variables if isinstance(module.variables, tuple) or \
                                                                        isinstance(module.variables, list)
                                                                     else module.variables()  for module \
                                                                                              in multi_agent.submodules]))
for var in variable_list:
    print(var.device, var.name)
# input("Press Enter to continue...")

# data = create_train_data(agent_list[0],
#                          multi_agent.wrap_policy(SmartCharger(0.5, num_actions, single_agent_time_step_spec), True),
#                          train_env,
#                          5)
st = time()
# strategy = tf.distribute.get_strategy()
# with strategy.scope():
#     multi_agent.train()
# return_list = []
# eval_policy = DummyV2G(0.5, num_actions, single_agent_time_step_spec)
# eval_policy = SmartCharger(0.5, num_actions, single_agent_time_step_spec)
# for i in range(5, 101, 5):
#     i /= 100
#     eval_policy.threshold = i
#     return_list.append((i, multi_agent.eval_policy([eval_policy]).numpy()))
# print(return_list)
multi_agent.train()
print(multi_agent.eval_policy())
et = time()
print('Expired time: {}'.format(et - st))
# print(multi_agent.returns)