import json
import math

import numpy as np
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.trajectories import TimeStep, Trajectory
import tensorflow as tf


# with open('data/vehicles_constant_shape.json') as file:
#     vehicles = VehicleDistributionListConstantShape(json.loads(file.read()))
#
# num_actions = 16
# single_agent_time_step_spec = TimeStep(
#     step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int64, minimum=0, maximum=2),
#     discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0.0, maximum=1.0),
#     reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
#     observation=tensor_spec.BoundedTensorSpec(shape=(34,), dtype=tf.float32, minimum=-1., maximum=1.))
#
# single_agent_action_spec = tensor_spec.BoundedTensorSpec(
#             shape=(), dtype=tf.int64, minimum=0, maximum=num_actions-1, name="action")
#
#
# epochs = 100
# policy = SmartCharger(0.5, num_actions, single_agent_time_step_spec)
# reward_function = rf.vanilla
# energy_curve_eval = EnergyCurve('data/randomized_data.csv', 'eval')
# env = TFPowerMarketEnv(energy_curve_eval,
#                        reward_function,
#                        1,
#                        [AVG_CHARGING_RATE * v * 1 for v in vehicles.avg_vehicles_list],
#                        False)
#
# kwargs = {
#     'time_step_spec': single_agent_time_step_spec,
#     'action_spec': single_agent_action_spec,
#     'q_network': q_net,
#     'target_q_network': target_q_net,
#     # 'optimizer': tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate,
#     #                                                                                   amsgrad=True),),
#     'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate),
#     'td_errors_loss_fn': common.element_wise_squared_loss,
#     # 'epsilon_greedy': epsilon_greedy,
#     'epsilon_greedy': None,
#     'boltzmann_temperature': 0.6,
#     'target_update_tau': 0.001,
#     'target_update_period': 1,
# }
# ckpt_dir = 'dump'
# coefficient_function = lambda x: tf.math.sin(math.pi / 6.0 * x) / 2.0 + 0.5
# create_single_agent(cls=DdqnAgent,
#                     ckpt_dir=ckpt_dir,
#                     vehicle_distribution=list(vehicles),
#                     buffer_max_size=MAX_BUFFER_SIZE,
#                     num_of_actions=num_actions,
#                     capacity_train_garage=100,
#                     capacity_eval_garage=200,
#                     name='Agent-',
#                     num_of_agents=1,
#                     coefficient_function=coefficient_function,
#                     **kwargs)
#
# def create_gaussian_outputs(action_num, value):
#     """
#     Creates an extrapolation of state action values based on the fact that according to the "smart policy" a
#     particular action is chosen, which means that it should have the highest value. Extrapolates the other values
#     considering a gaussian distribution which is maximized at the value of the state action chosen and has a
#     standard deviation of 1 (here the x values of the distribution are the different actions (0 - num_actions)
#     """
#     gaussian = lambda x: tf.math.minimum(0., 2. * value) + tf.abs(value) * \
#                          tf.math.exp(-1.0 * (x - action_num) ** 2 / (2 * 1 ** 2))
#     return tf.vectorized_map(lambda t: gaussian(t),
#                              tf.range(num_actions, dtype=tf.float32))
#
# observation_var = tf.Variable(shape=[24] + single_agent_time_step_spec.observation.shape,
#                               dtype=single_agent_time_step_spec.observation.dtype)
# rewards_var = tf.Variable(shape=[24] + [num_actions],
#                           dtype=single_agent_time_step_spec.reward.dtype)
# index = tf.Variable(0)
# def log_obs_rewards(trajectory: Trajectory):
#     if trajectory.is_last():
#         index.assign(0)
#         ...
#     else:
#         id = tf.squeeze(trajectory.policy_info)
#         obs = tf.concat((trajectory.observation, tf.gather(agent._private_observations, id)), axis=0)
#         action = tf.gather(agent._private_actions, id)
#         observation_var.scatter_nd_update(tf.expand_dims(index, 0), trajectory.observation)
#         rewards_var.scatter_nd_update(tf.expand_dims(index, 0), [create_gaussian_outputs(agent ,trajectory.reward)])
#         index.assign_add(1)
#
# driver = TFDriver(env,
#                   policy,
#                   [log_obs_rewards],
#                   max_episodes=epochs)

def create_train_data(agent, policy, env, epochs):
    num_actions = agent.collect_policy.action_spec.maximum + 1

    def create_gaussian_outputs(action_num, value):
        """
        Creates an extrapolation of state action values based on the fact that according to the "smart policy" a
        particular action is chosen, which means that it should have the highest value. Extrapolates the other values
        considering a gaussian distribution which is maximized at the value of the state action chosen and has a
        standard deviation of 1 (here the x values of the distribution are the different actions (0 - num_actions)
        """
        gaussian = lambda x: tf.math.minimum(0., 2. * value) + tf.abs(value) * \
                             tf.math.exp(-1.0 * (x - action_num) ** 2 / (2 * 1 ** 2))
        return tf.vectorized_map(lambda t: gaussian(t),
                                 tf.range(num_actions, dtype=tf.float32))
    index = tf.Variable(0)
    final_index = tf.Variable(0)
    observation_var = tf.Variable(tf.zeros(shape=[24] + agent.time_step_spec.observation.shape,
                                           dtype=agent.time_step_spec.observation.dtype))
    action_var = tf.Variable(tf.zeros(shape=[24] + agent.action_spec.shape,
                                      dtype=agent.time_step_spec.reward.dtype))
    rewards_var = tf.Variable(tf.zeros(shape=[24] + agent.time_step_spec.reward.shape,
                                       dtype=agent.time_step_spec.reward.dtype))
    final_tensor = tf.Variable(tf.zeros(shape=[epochs * 100 * 24,
                                               agent.time_step_spec.observation.shape.num_elements() + num_actions],
                                        dtype=tf.float32))
    def gaussian_callback(trajectory: Trajectory):
        if trajectory.is_boundary():
            total_rewards = tf.math.cumsum(rewards_var.value(), axis=0, reverse=True)
            gaussian_data = tf.map_fn(lambda t: create_gaussian_outputs(t[0], t[1]),
                                      tf.stack((action_var, total_rewards), axis=1),
                                      parallel_iterations=24)
            final_tensor.scatter_nd_update(tf.expand_dims(tf.range(final_index, final_index + 24, dtype=tf.int32),
                                                          axis=1),
                                           tf.concat((observation_var, gaussian_data), axis=1))
            index.assign(0)
            final_index.assign_add(24)
        else:
            id = tf.squeeze(trajectory.policy_info)
            obs = tf.concat((trajectory.observation,
                             tf.expand_dims(tf.gather(agent._private_observations, id), axis=0)), axis=1)
            action = tf.cast(tf.gather(agent._private_actions, id), tf.float32)
            update_id = tf.expand_dims(tf.expand_dims(index, axis=0), axis=1)
            observation_var.scatter_nd_update(update_id, obs)
            rewards_var.scatter_nd_update(update_id, trajectory.reward[..., 0])
            action_var.scatter_nd_update(update_id, tf.expand_dims(action, axis=0))
            index.assign_add(1)
    driver = TFDriver(env,
                      policy,
                      [gaussian_callback],
                      max_episodes=100)
    for _ in range(5):
        env.hard_reset()
        driver.run(env.reset())

    with open('colab_data_output_16.csv', "w") as file:
        np.savetxt(file, final_tensor.numpy(), delimiter=',')

