import numpy as np
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.trajectories import TimeStep, Trajectory
import tensorflow as tf

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
                             tf.math.exp(-0.5 * ((x - action_num) / 0.5) ** 2)
                             # tf.math.exp(-1.0 * (x - action_num) ** 2 / (2 * 0.5 ** 2))
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
    with open('colab_data_output_8.csv', "w") as file:
        np.savetxt(file, final_tensor.numpy(), delimiter=',')



