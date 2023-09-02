import json
import math
import sys

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import TimeStep, Trajectory

from app.abstract.tf_single_agent_single_model import create_single_agent
from app.models.tf_energy_3 import EnergyCurve
from app.models.tf_pwr_env_5 import TFPowerMarketEnv
from app.policies.multiple_tf_agents_single_model import MultiAgentSingleModelPolicy
from app.policies.tf_reward_functions import vanilla
from app.policies.tf_smarter_charger import SmartCharger
from app.utils import VehicleDistributionListConstantShape, calculate_avg_distribution_constant_shape
from config import AVG_CHARGING_RATE, MAX_BUFFER_SIZE

tf.config.run_functions_eagerly(False)

try:
    scale_upwards = float(sys.argv[1])
except:
    scale_upwards = 0.0
try:
    epochs = int(sys.argv[2])
except:
    epochs = 10
try:
    offset = bool(sys.argv[3])
except:
    offset = False
try:
    dir = sys.argv[4]
except:
    dir = 'colab_data_output_8_35.csv' if not offset else 'colab_data_output_8_35_offset.csv'
print(
    'Creating data with parameters: scaling: {}, epochs: {}, offset: {}, '.format(1.0 + scale_upwards, epochs, offset))


def create_train_data(agent, policy, env, epochs):
    num_actions = agent.collect_policy.action_spec.maximum + 1

    def create_gaussian_outputs(action_num, value):
        """
        Creates an extrapolation of state action values based on the fact that according to the "smart policy" a
        particular action is chosen, which means that it should have the highest value. Extrapolates the other values
        considering a gaussian distribution which is maximized at the value of the state action chosen and has a
        standard deviation of 2 (here the x values of the distribution are the different actions (0 - num_actions)
        """
        gaussian = lambda x: tf.math.minimum(0., 2. * value) + tf.abs(value) * \
                             tf.math.exp(-0.5 * ((x - action_num) / 2) ** 2)
        # tf.math.exp(-1.0 * (x - action_num) ** 2 / (2 * 0.5 ** 2))
        return tf.vectorized_map(lambda t: gaussian(t),

                                 tf.range(num_actions, dtype=tf.float32))

    index = tf.Variable(0)
    final_index = tf.Variable(0)
    observation_var = tf.Variable(tf.zeros(shape=[24] + agent.time_step_spec.observation.shape,
                                           dtype=agent.time_step_spec.observation.dtype))
    action_var = tf.Variable(tf.zeros(shape=[24] + agent.action_spec.shape,
                                      dtype=tf.int64))
    rewards_var = tf.Variable(tf.zeros(shape=[24] + agent.time_step_spec.reward.shape,
                                       dtype=agent.time_step_spec.reward.dtype))
    final_tensor = tf.Variable(tf.zeros(shape=[epochs * 100 * 24,
                                               agent.time_step_spec.observation.shape.num_elements() + num_actions],
                                        dtype=tf.float32))

    def gaussian_callback(trajectory: Trajectory):
        if trajectory.is_boundary():
            total_rewards = tf.math.cumsum(rewards_var.value(), axis=0, reverse=True)
            gaussian_data = tf.map_fn(lambda t: create_gaussian_outputs(t[0], t[1]),
                                      tf.stack((tf.cast(action_var, tf.float32), total_rewards), axis=-1),
                                      parallel_iterations=24)
            final_tensor.scatter_nd_update(tf.expand_dims(tf.range(final_index, final_index + 24, dtype=tf.int32),
                                                          axis=1),
                                           tf.concat((observation_var, gaussian_data), axis=1))
            index.assign(0)
            final_index.assign_add(24)
        else:
            # id = tf.squeeze(trajectory.policy_info)
            # obs = tf.concat((trajectory.observation,
            #                  tf.expand_dims(tf.gather(agent._private_observations, id), axis=0)), axis=1)
            # action = tf.cast(tf.gather(agent._private_actions, id), tf.float32)
            obs = trajectory.observation[0]
            # update_id = tf.expand_dims(tf.expand_dims(index, axis=0), axis=1)
            update_id = tf.reshape(index, [1, 1])
            observation_var.scatter_nd_update(update_id, [tf.squeeze(obs)])
            rewards_var.scatter_nd_update(update_id, trajectory.reward[..., 0])
            action_var.scatter_nd_update(update_id, trajectory.action[..., 0])
            index.assign_add(1)

    driver = TFDriver(env,
                      policy,
                      [lambda traj: gaussian_callback(policy.get_last_trajectory(traj))],
                      max_episodes=100)
    for _ in range(5):
        env.hard_reset()
        driver.run(env.reset())
    if scale_upwards > 0.0:
        final_tensor.assign_add(
            tf.math.abs(final_tensor) * tf.constant((tf.shape(final_tensor)[-1] - 8) * [0.0] + [scale_upwards] * 8))
    with open(dir, "w") as file:
        np.savetxt(file, final_tensor.numpy(), delimiter=',')


with open('data/vehicles_constant_shape.json') as file:
    vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))
with open('data/vehicles_constant_shape_offset.json') as file:
    offset_vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))

spec = TimeStep(
    step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int64, minimum=0, maximum=2),
    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0.0, maximum=1.0),
    reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
    observation=tensor_spec.BoundedTensorSpec(shape=(35,), dtype=tf.float32, minimum=-1., maximum=1.))
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
q_net = sequential.Sequential(layers_list)
new_q_net = q_net.copy(name='Agent_{}_QNetwork'.format(1))
new_target_q_net = q_net.copy(name='Agent_{}_TargetQNetwork'.format(1))
kwargs = {
    'time_step_spec': spec,
    'action_spec': single_agent_action_spec,
    # 'q_network': q_net,
    # 'q_network': q_net if not offset else offset_q_net,
    'q_network': new_q_net,
    'target_q_network': new_target_q_net,
    # 'target_q_network': target_q_net,
    # 'target_q_network': target_q_net if not offset else offset_target_q_net,
    # 'optimizer': tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate,
    #                                                                                   amsgrad=True),),
    'optimizer': tf.keras.optimizers.Adam(learning_rate=3e-2),
    # 'td_errors_loss_fn': common.element_wise_squared_loss,
    # 'epsilon_greedy': 0.2,
    'epsilon_greedy': None,
    'boltzmann_temperature': 0.9,
    'target_update_tau': 0.1,
    'target_update_period': 2400,
}

coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5
offset_coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * (x + 3.0)) / 2.0 + 0.5
agent_list = [[]]
agent_list[0] = create_single_agent(cls=DdqnAgent,
                                    ckpt_dir='dump',
                                    # vehicle_distribution=list(offset_vehicles),
                                    vehicle_distribution=list(vehicles) if not offset else offset_vehicles,
                                    buffer_max_size=MAX_BUFFER_SIZE,
                                    num_of_actions=num_actions,
                                    capacity_train_garage=100,
                                    capacity_eval_garage=100,
                                    name='Agent-' + str(1),
                                    # num_of_agents=1,
                                    # coefficient_function=offset_coefficient_function,
                                    coefficient_function=coefficient_function if not offset else offset_coefficient_function,
                                    **kwargs)
energy_curve_train = EnergyCurve('data/data_sorted_by_date.csv', 'train')
collect_avg_vehicles_list = tf.constant([0.0] * 24)
days = 500
for agent in agent_list:
    collect_avg_vehicles_list += calculate_avg_distribution_constant_shape(days,
                                                                           agent.train_vehicles_generator)
train_env = TFPowerMarketEnv(spec,
                             # env_action_spec,
                             energy_curve_train,
                             vanilla,
                             1,
                             [AVG_CHARGING_RATE * v for v in collect_avg_vehicles_list.numpy()],
                             True)

create_train_data(agent_list[0],
                  MultiAgentSingleModelPolicy(SmartCharger(0.05, num_actions, spec),
                                              [agent_list[0]],
                                              train_env.time_step_spec(),
                                              train_env.action_spec(),
                                              (),
                                              (),
                                              tf.int64,
                                              True),
                  train_env,
                  epochs)
