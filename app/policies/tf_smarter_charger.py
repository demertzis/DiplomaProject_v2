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
        # current_price = observation[..., 0]
        start = self._num_of_multi_agent_observation_elements
        max_coefficient, threshold_coefficient, min_coefficient = tf.unstack(observation[..., start:start + 3], axis=-1)

        price_tensor = observation[..., :12]
        positive_indexes = tf.math.greater_equal(price_tensor, 0.0)
        positive_values = tf.reduce_sum(tf.where(positive_indexes, 1.0, 0.0))
        # mean_current_price = tf.math.divide_no_nan(tf.reduce_sum(tf.where(price_tensor >= 0.0, price_tensor, 0.0)),
        #                                            positive_values)
        min_price = tf.reduce_min(tf.where(positive_indexes, price_tensor, 1.0))
        max_price = tf.reduce_max(tf.where(positive_indexes, price_tensor, 1.0))
        # current_price = tf.math.divide_no_nan(observation[..., 0] - min_price, mean_current_price)
        current_price = tf.math.divide_no_nan(observation[..., 0] - min_price, max_price - min_price)
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        if coefficient_step == 0:
            return PolicyStep(action=(self.actions_length - 1) * tf.ones_like(coefficient_step, dtype=tf.int64))
        good_price = current_price < self.threshold

        # coefficient = threshold_coefficient

        if good_price:
            coefficient = (threshold_coefficient - max_coefficient) * (current_price / self.threshold) + max_coefficient
        else:
            # coefficient = min_coefficient - (min_coefficient - threshold_coefficient) * math.e ** (
            #     1.0 - current_price / self.threshold
            # )
            price_deviation = tf.math.divide_no_nan((current_price - self._threshold), (1.0 - self._threshold))
            # price_deviation = current_price
            # price_deviation = (current_price - self._threshold) / (1.0 - self._threshold)
            coefficient = (1.0 - tf.clip_by_value(price_deviation * tf.math.exp(0.05 * price_deviation),
                                                  0.0,
                                                  1.0)) * \
                          (threshold_coefficient - min_coefficient) + \
                          min_coefficient
            # price_deviation = tf.clip_by_value(3.0 * tf.math.sigmoid(5 * price_deviation) - 0.5, 0.0, 1.0)
            # coefficient = min_coefficient + (max_coefficient - min_coefficient) * \
            #               (1.0 - tf.clip_by_value(tf.math.log(price_deviation /(1.0 - price_deviation)  + 0.5), 0.0, 1.0))
        # else:
        #     coefficient = min_coefficient - (min_coefficient - threshold_coefficient) * math.e ** (
        #         1.0 - current_price / self.threshold
        #     )
        action = tf.cast(tf.round((coefficient - min_coefficient) / coefficient_step), dtype=tf.int64)
        return PolicyStep(action=action)