import numpy as np
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
import tensorflow as tf




# def _preprocess_sequence(self, experience: types.NestedTensor):
#     parking_observation = self._private_observations[experience.policy_info]
#     augmented_observation = tf.constant([*parking_observation[:3],
#                                          *experience.observation[:12],
#                                          *parking_observation[3:],
#                                          experience.observation[12]],
#                                         shape=self._training_data_spec)
#     agent_reward = experience.reward[self._agent_id: self._agent_id + 1]
#     return experience.replace(observation=augmented_observation, policy_info=(), reward=agent_reward)
def _preprocess_sequence(self, experience: types.NestedTensor):
    # if self._num_outer_dims != 1:
    #     raise Exception('Does not support stateful agents currently')
    parking_obs = tf.gather(self._private_observations, experience.policy_info)
    augmented_obs = tf.concat(
        (tf.concat((tf.concat((parking_obs[:, :, :3], experience.observation[:, :, :12]), axis=2),
                    parking_obs[:, :, 3:]), axis=2),
         experience.observation[:, :, 12]), axis=2)
    agent_reward = experience.reward[:, :, self._agent_id: self._agent_id + 1]
    return experience.replace(observation=augmented_obs, policy_info=(), reward=agent_reward)


experience = Trajectory(
    step_type=0,
    observation=tf.constant([*[0.0]*10, *[0.4]*2, *[1.0]], shape=(1,13)),
    action=tf.constant
    (np.array((199.9, 12.3)), shape=(1,2)),
    policy_info=tf.constant((0)
                        ),
    next_step_type=1,
    reward=tf.constant((1.0,2.0), shape=(1,2)),
    discount=0.9
)

class Foo:
    _private_observations=tf.constant([[1.0] * 21, [2.3] * 21])
    _agent_id = 1
    _training_data_spec = (34,)

self = Foo()

print(_preprocess_sequence(self, experience))

