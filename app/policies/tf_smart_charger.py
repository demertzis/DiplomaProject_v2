import math
from typing import Optional

import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class SmartCharger:

    def __init__(self, threshold, actions_length, time_step_spec):
        self._threshold = tf.Variable(tf.constant(threshold, dtype=tf.float32))
        self.actions_length = actions_length
        self._input_tensor_spec = time_step_spec.observation
        self._num_of_multi_agent_observation_elements = time_step_spec.observation.shape[0] - 21

    @property
    def input_tensor_spec(self):
        return self._input_tensor_spec

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        self._threshold.assign(tf.constant(value, tf.float32))

    def action(self, timestep: TimeStep, policy_state = (), seed: Optional[types.Seed] = None) -> PolicyStep:
        nest_utils.assert_matching_dtypes_and_inner_shapes(
            timestep.observation,
            self.input_tensor_spec,
            allow_extra_fields=True,
            caller=self,
            tensors_name="`timestep`",
            specs_name="`input_tensor_spec`")
        observation = timestep.observation
        current_price = observation[..., 0]
        start = self._num_of_multi_agent_observation_elements
        max_coefficient, threshold_coefficient, min_coefficient = tf.unstack(observation[..., start:start + 3], axis=-1)

        # max_coefficient = observation[13:14]
        # threshold_coefficient = observation[14:15]
        # min_coefficient = observation[15:16]
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        if coefficient_step == 0:
            return PolicyStep(action=(self.actions_length - 1) * tf.ones_like(coefficient_step, dtype=tf.int64))
        good_price = current_price < self.threshold
        if good_price:
            coefficient = (threshold_coefficient - max_coefficient) * (current_price / self.threshold) + max_coefficient
        else:
            coefficient = min_coefficient - (min_coefficient - threshold_coefficient) * math.e ** (
                1.0 - current_price / self.threshold
            )
        action = tf.cast(tf.round((coefficient - min_coefficient) / coefficient_step), dtype=tf.int64)
        return PolicyStep(action=action)


