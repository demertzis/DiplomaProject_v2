import json
import math
import sys
from time import time

import tensorflow as tf
from keras.src.optimizers.schedules import ExponentialDecay
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep

import app.policies.tf_reward_functions as rf
import config
from app.abstract.tf_single_agent_single_model import create_single_agent
from app.abstract.utils import RandomDistributionTrainScheduler
from app.models.tf_energy_3 import EnergyCurve
from app.models.tf_pwr_env_5 import TFPowerMarketEnv
from app.policies.multiple_tf_agents_single_model import MultipleAgents
from app.policies.tf_smarter_charger import SmartCharger
from app.utils import VehicleDistributionListConstantShape, calculate_avg_distribution_constant_shape
# from config import NUMBER_OF_AGENTS, MAX_BUFFER_SIZE, AVG_CHARGING_RATE
from config import MAX_BUFFER_SIZE, AVG_CHARGING_RATE

    # from data_creation_3 import create_train_data
tf.config.run_functions_eagerly(config.EAGER_EXECUTION)
try:
    argument_num_of_agents = int(sys.argv[1])
except:
    argument_num_of_agents = config.NUMBER_OF_AGENTS
print('Ploting {} Agents'.format(argument_num_of_agents))
NUMBER_OF_AGENTS = argument_num_of_agents
# tf.debugging.enable_check_numerics()
reward_function_array = [rf.vanilla,
                         rf.punishing_uniform,
                         rf.punishing_non_uniform_individually_rational,
                         rf.punishing_non_uniform_non_individually_rational,
                         rf.new_reward_halfway,
                         rf.new_reward_buyers_biased,
                         rf.new_reward_sellers_biased,
                         rf.new_reward_proportional_punishing,
                         rf.new_reward_equal,
                         ]
try:
    reward_function = reward_function_array[int(sys.argv[2])]
except:
    reward_function = reward_function_array[0]

if NUMBER_OF_AGENTS == 1:
    try:
        single_agent_offset = bool(int(sys.argv[3]))
    except:
        single_agent_offset = False

if NUMBER_OF_AGENTS == 1:
    print("Train run: {} agents, reward function = {}, single agent offset = {}".format(NUMBER_OF_AGENTS,
                                                                                        reward_function.__name__,
                                                                                        single_agent_offset))
else:
    print("Train run: {} agents, reward function = {}".format(NUMBER_OF_AGENTS, reward_function.__name__))

# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
with open('data/vehicles_constant_shape.json') as file:
    vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))
with open('data/vehicles_constant_shape_offset.json') as file:
    offset_vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))

single_agent_time_step_spec = TimeStep(
    step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int64, minimum=0, maximum=2),
    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0.0, maximum=1.0),
    reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
    observation=tensor_spec.BoundedTensorSpec(shape=(35,), dtype=tf.float32, minimum=-1.,
                                              maximum=1.))  # was 33 but added 2 elements (intra day pricing and time of day (1.0 at 0, 0.0 at 24)

num_actions = 8
single_agent_action_spec = tensor_spec.BoundedTensorSpec(
    shape=(), dtype=tf.int64, minimum=0, maximum=num_actions - 1, name="action")

layers_list = \
    [
        # tf.keras.layers.Dense(units=35, activation="elu"),
        tf.keras.layers.Dense(units=64, activation="elu"),
        tf.keras.layers.Dense(units=128, activation="elu"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            num_actions,
        ),
        tf.keras.layers.Activation('linear', dtype=tf.float32)
    ]


def load_pretrained_model(model_dir):
    keras_model = tf.keras.models.load_model(model_dir)
    layers_list = []
    i = 0
    while True:
        try:
            layers_list.append(keras_model.get_layer(index=i))
        except IndexError:
            print('{0}: Total number of layers in neural network: {1}'.format('q-net', i))
            break
        except ValueError:
            print('{0}: Total number of layers in neural network: {1}'.format('offset-q-net', i))
            break
        else:
            i += 1

    temp_model = tf.keras.Sequential(layers_list)
    # temp_model.build(input_shape=(1,35))
    # temp_target_model = tf.keras.models.clone_model(temp_model)

    q_net = sequential.Sequential(temp_model.layers, name='QNetwork')
    # target_q_net = sequential.Sequential(temp_target_model.layers, name='TargetQNetwork')
    # target_q_net = q_net.copy(name='TargetQNetwork')
    # return q_net, target_q_net
    return q_net


model_dir = 'pretrained_networks/'
best_model_dir = 'pretrained_networks/best_models/'
# best_model_dir = 'pretrained_networks/new_models/'
try:
    q_net = load_pretrained_model(best_model_dir + 'model_output_8_35.keras')
except OSError:
    q_net = load_pretrained_model(model_dir + 'model_output_8_35.keras')
try:
    offset_q_net = load_pretrained_model(best_model_dir + 'model_output_8_35_offset.keras')
except OSError:
    offset_q_net = load_pretrained_model(model_dir + 'model_output_8_35_offset.keras')

learning_rate = 3e-4
reward_name = reward_function.__name__ if not config.PLOT_PRETRAINED_NETWORK else 'pretrained_policy'

ckpt_dir = '/'.join(['checkpoints_2',
                     str(NUMBER_OF_AGENTS) +
                     '_AGENTS',
                     reward_name])

if NUMBER_OF_AGENTS == 1 and single_agent_offset:
    ckpt_dir += '_OFFSET'
coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5
offset_coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * (x + 3.0)) / 2.0 + 0.5
agent_list = []
# offset = False

gpus = tf.config.list_physical_devices('GPU')
# successful_gpu_division = False
# if gpus:
#     try:
#         gpu_mem = 15 * 1024 - tf.config.experimental.get_memory_info('GPU:0')
#         num_devices = NUMBER_OF_AGENTS + 1
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=(gpu_mem - 1024) / num_devices)] * (num_devices))
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)
#     successful_gpu_division = True
for i in range(NUMBER_OF_AGENTS):
    with tf.device(f'GPU:0' if gpus else 'CPU:0'):
        if i % 3 == 2:
            offset = True
        else:
            offset = False
        if NUMBER_OF_AGENTS == 1:
            offset = single_agent_offset
        if config.START_FROM_SCRATCH:
            layers = []
            for units in [40, 80, num_actions]:
                layers.append(tf.keras.layers.Dense(
                    units,
                    activation=tf.keras.activations.elu,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0, mode='fan_in', distribution='truncated_normal')))

            new_q_net = sequential.Sequential(layers, name='Agent_{}_QNetwork'.format(i))
            new_target_q_net = new_q_net.copy(name='Agent_{}_TargetQNetwork'.format(i))
        else:
            new_q_net = (offset_q_net if offset else q_net).copy(name='Agent_{}_QNetwork'.format(i))
            new_target_q_net = new_q_net.copy(name='Agent_{}_TargetQNetwork'.format(i))
        kwargs = {
            'time_step_spec': single_agent_time_step_spec,
            'action_spec': single_agent_action_spec,
            # 'q_network': q_net,
            # 'q_network': q_net if not offset else offset_q_net,
            'q_network': new_q_net,
            'target_q_network': new_target_q_net,
            # 'target_q_network': target_q_net,
            # 'target_q_network': target_q_net if not offset else offset_target_q_net,
            # 'optimizer': tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate,
            #                                                                                   amsgrad=True),),
            'optimizer': tf.keras.optimizers.AdamW(learning_rate=learning_rate),
            # 'td_errors_loss_fn': common.element_wise_squared_loss,
            # 'epsilon_greedy': 0.2,single_agent_offset
            'epsilon_greedy': None,
            'boltzmann_temperature': 0.9,
            'target_update_tau': 0.1,
            'target_update_period': 2400,
        }
        agent_list.append(create_single_agent(cls=DdqnAgent,
                                              ckpt_dir=ckpt_dir,
                                              # vehicle_distribution=list(offset_vehicles),
                                              vehicle_distribution=list(vehicles) if not offset else offset_vehicles,
                                              buffer_max_size=MAX_BUFFER_SIZE,
                                              num_of_actions=num_actions,
                                              capacity_train_garage=100,
                                              capacity_eval_garage=100,
                                              name='Agent-' + str(i + 1),
                                              # num_of_agents=NUMBER_OF_AGENTS,
                                              # coefficient_function=offset_coefficient_function,
                                              coefficient_function=coefficient_function if not offset else offset_coefficient_function,
                                              **kwargs))

collect_avg_vehicles_list = tf.constant([0.0] * 24)
days = 500
for agent in agent_list:
    collect_avg_vehicles_list += calculate_avg_distribution_constant_shape(days,
                                                                           agent.train_vehicles_generator)
# collect_avg_vehicles_list = collect_avg_vehicles_list.numpy()

eval_avg_vehicle_list = tf.constant([0.0] * 24)
if len(agent_list) == 1:
    eval_avg_vehicle_list += vehicles.avg_vehicles_list if not single_agent_offset else offset_vehicles.avg_vehicles_list
else:
    for i in range(len(agent_list)):
        eval_avg_vehicle_list += vehicles.avg_vehicles_list if i % 3 != 2 else offset_vehicles.avg_vehicles_list
    # eval_avg_vehicle_list += offset_vehicles.avg_vehicles_list

with tf.device(f'GPU:0' if gpus else 'CPU:0'):
    energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
    energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')

    spec = single_agent_time_step_spec
    train_env = TFPowerMarketEnv(spec,
                                 # env_action_spec,
                                 energy_curve_train,
                                 reward_function,
                                 NUMBER_OF_AGENTS,
                                 [AVG_CHARGING_RATE * v for v in collect_avg_vehicles_list.numpy()],
                                 True)
    eval_env = TFPowerMarketEnv(spec,
                                # env_action_spec,
                                energy_curve_eval,
                                reward_function,
                                NUMBER_OF_AGENTS,
                                [AVG_CHARGING_RATE * v for v in eval_avg_vehicle_list.numpy()],
                                False)

    base_lr = 1e-4
    lr_list = [ExponentialDecay(1e-4, 1, 0.8), 4e-7, 1e-7]
    epochs_per_agent = [[0, 10]]
    for i in range(1, len(agent_list)):
        temp_list = epochs_per_agent + [[i, 1]]
        epochs_per_agent.reverse()
        epochs_per_agent = temp_list + epochs_per_agent

    train_scheduler = RandomDistributionTrainScheduler(len(agent_list), ExponentialDecay(6e-7, 1, 0.9), 3e-8, 4, 1)
    multi_agent = MultipleAgents(train_env=train_env,
                                 eval_env=eval_env,
                                 agents_list=agent_list,
                                 ckpt_dir=ckpt_dir,
                                 initial_collect_policy=None,
                                 train_scheduler=train_scheduler)

# print(multi_agent.eval_policy(best=True))
# plot_actions = True
# plot_actions = False
# last_folder = ['actions'] if plot_actions else ['rewards']
last_folder = ['actions']
if NUMBER_OF_AGENTS == 1:
    offset_string = '_offset' if single_agent_offset else ''
    single_reward = 'vanilla' if not config.PLOT_PRETRAINED_NETWORK else 'pretrained_policy'
    plot_filename = '/'.join(['plots',
                              str(NUMBER_OF_AGENTS) + '_Agent',
                              single_reward + offset_string] +
                             last_folder)
else:
    plot_filename ='/'.join(['plots',
                             str(NUMBER_OF_AGENTS) + '_Agents',
                             reward_name] +
                            last_folder)
multi_agent.plot_actions(filename=plot_filename,
                         best=True,
                         actions=True)

# plot_actions = True
# plot_actions = False
# last_folder = ['actions'] if plot_actions else ['rewards']
last_folder = ['rewards']
if NUMBER_OF_AGENTS == 1:
    offset_string = '_offset' if single_agent_offset else ''
    single_reward = 'vanilla' if not config.PLOT_PRETRAINED_NETWORK else 'pretrained_policy'
    plot_filename = '/'.join(['plots',
                              str(NUMBER_OF_AGENTS) + '_Agent',
                              single_reward + offset_string] +
                             last_folder)
else:
    plot_filename ='/'.join(['plots',
                             str(NUMBER_OF_AGENTS) + '_Agents',
                             reward_name] +
                            last_folder)
multi_agent.plot_actions(filename=plot_filename,
                         best=True,
                         actions=False)