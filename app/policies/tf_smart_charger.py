import numpy as np
import math
import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep


class DummyV2G:
    actions_length = 21

    def __init__(self, threshold):
        self.threshold = threshold

    def action(self, timestep: TimeStep) -> PolicyStep:
        observation = tf.squeeze(timestep.observation)
        current_price = observation[0]
        max_coefficient, threshold_coefficient, min_coefficient = observation[13:16]
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        if coefficient_step == 0:
            return PolicyStep(action=tf.constant([self.actions_length - 1], dtype=np.int32))
        good_price = current_price < self.threshold
        if good_price:
            coefficient = (threshold_coefficient - max_coefficient) * (current_price / self.threshold) + max_coefficient
        else:
            coefficient = min_coefficient - (min_coefficient - threshold_coefficient) * math.e ** (
                1.0 - current_price / self.threshold
            )
        return PolicyStep(
            action=tf.constant(np.array([tf.round((coefficient - min_coefficient) / coefficient_step)], dtype=np.int32))
        )


