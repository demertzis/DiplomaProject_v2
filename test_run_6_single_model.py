import json
import math
import sys
from time import time

import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep

import app.policies.tf_reward_functions as rf
import config
from app.abstract.tf_single_agent_single_model import create_single_agent
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
NUMBER_OF_AGENTS = argument_num_of_agents
# tf.debugging.enable_check_numerics()
reward_function_array = [rf.vanilla,
                         rf.punishing_uniform,
                         rf.punishing_non_uniform_individually_rational,
                         rf.punishing_non_uniform_non_individually_rational,]
try:
    reward_function = reward_function_array[int(sys.argv[2])]
except:
    reward_function = reward_function_array[0]

try:
    if NUMBER_OF_AGENTS == 1:
        single_agent_offset = bool(int(sys.argv[3]))
except:
    single_agent_offset = False

if NUMBER_OF_AGENTS == 1:
    print("Train run: {} agents, reward function = {}, single agent offset = {}".format(NUMBER_OF_AGENTS,
                                                                                        reward_function.__name__,
                                                                                        single_agent_offset))
else:
    print("Train run: {} agents, reward function = {}".format(NUMBER_OF_AGENTS, reward_function.__name__))


tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
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
try:
    q_net = load_pretrained_model(best_model_dir + 'model_output_8_35.keras')
except OSError:
    q_net = load_pretrained_model(model_dir + 'model_output_8_35.keras')
try:
    offset_q_net = load_pretrained_model(best_model_dir + 'model_output_8_35_offset.keras')
except OSError:
    offset_q_net = load_pretrained_model(model_dir + 'model_output_8_35_offset.keras')

# q_net = sequential.Sequential(layers_list)
# q_net.build(input_shape=(1,35))
# offset_q_net = sequential.Sequential(layers_list)
# offset_q_net.build(input_shape=(1,35))


learning_rate = 3e-4
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

# reward_function = rf.vanilla
# reward_function = rf.punishing_uniform
# reward_function = rf.punishing_non_uniform_non_individually_rational
# reward_function = rf.punishing_non_uniform_individually_rational
reward_name = reward_function.__name__

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
successful_gpu_division = False
if gpus:
    try:
        gpu_mem = 15 * 1024 - tf.config.experimental.get_memory_info('GPU:0')
        num_devices = NUMBER_OF_AGENTS + 1
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=(gpu_mem - 1024) / num_devices)] * (num_devices))
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    successful_gpu_division = True
for i in range(NUMBER_OF_AGENTS):
    with tf.device(f'GPU:{i}' if successful_gpu_division else 'CPU:0'):
        if i % 3 == 2:
            offset = True
        else:
            offset = False
        if NUMBER_OF_AGENTS == 1:
            offset = single_agent_offset
        new_q_net = (offset_q_net if offset else q_net).copy(name='Agent_{}_QNetwork'.format(i))
        new_target_q_net = (offset_q_net if offset else q_net).copy(name='Agent_{}_TargetQNetwork'.format(i))
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
            'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

with tf.device(f'GPU:{NUMBER_OF_AGENTS}' if successful_gpu_division else 'CPU:0'):
    energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
    energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')

    spec = single_agent_time_step_spec
    # env_time_step_spec = TimeStep(step_type=tensor_spec.add_outer_dim(spec.step_type, 1),
    #                               discount=tensor_spec.add_outer_dim(spec.discount, 1),
    #                               reward=tensor_spec.add_outer_dim(spec.reward, NUMBER_OF_AGENTS),
    #                               observation=tensor_spec.BoundedTensorSpec(shape=(13,),
    #                                                                         dtype=tf.float32,
    #                                                                         minimum=-1.,
    #                                                                         maximum=1.))
    # env_action_spec = tensor_spec.TensorSpec(shape=(NUMBER_OF_AGENTS,), dtype=tf.float32, name="action")
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

# for i, agent in enumerate(agent_list):
#     new_model = tf.keras.models.Sequential(agent._q_network.layers)
#     new_model.build((1,35))
#     new_model.save('pretrained_networks/new_models/model_output_8_agent' + str(i) + '.keras')
#     # new_model.save('pretrained_networks/model_output_8.keras')

# def create_data():
#     vehicle_tensor = tf.zeros(shape=(0, 100,3), dtype=tf.float32)
#     for i in range(40 * 24):
#         vehicle_tensor = tf.concat((vehicle_tensor, tf.expand_dims(agent_list[0].train_vehicles_generator(tf.constant(i % 24, tf.int64)), axis=0)), axis=0)
#     return vehicle_tensor.numpy().round(2)
# data = create_data()
# with open('data/vehicles_constant_shape_offset.json', 'w') as f:
#     json.dump(data.tolist(), f)
multi_agent = MultipleAgents(train_env,
                             eval_env,
                             agent_list,
                             ckpt_dir, )
# SmartCharger(0.5, num_actions, single_agent_time_step_spec))

# variable_list = list(itertools.chain.from_iterable([module.variables if isinstance(module.variables, tuple) or \
#                                                                         isinstance(module.variables, list)
#                                                                      else module.variables()  for module \
#                                                                                               in multi_agent.submodules]))
# for var in variable_list:
#     print(var.device, var.name)

# data = create_train_data(agent_list[0],
#                          MultiAgentSingleModelPolicy(SmartCharger(0.4, num_actions, single_agent_time_step_spec),
#                                                      [agent_list[0]],
#                                                      train_env.time_step_spec(),
#                                                      train_env.action_spec(),
#                                                      (),
#                                                      (),
#                                                      tf.int64,
#                                                      True),
#                          train_env,
#                          5)
if config.USE_JIT:
    st = time()
    multi_agent.train()
    print('Expired time: {}'.format(time() - st))
else:
    return_list = []
    # eval_policy = DummyV2G(0.5, num_actions, single_agent_time_step_spec)
    eval_policy = [SmartCharger(0.5, num_actions, single_agent_time_step_spec) for _ in multi_agent._agent_list]
    # for i in [1] + list(range(5, 101, 5)):
    #     i /= 100
    #     for policy in eval_policy:
    #         policy.threshold = i
    #     return_list.append((i, multi_agent.eval_policy(eval_policy).numpy()))
    # print(return_list)
    print(multi_agent.eval_policy())
    # input("Press Enter to continue...")
    st = time()
    multi_agent.train()
    print('Expired time: {}'.format(time() - st))
