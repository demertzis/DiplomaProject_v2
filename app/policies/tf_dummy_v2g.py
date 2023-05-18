import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import nest_utils


class DummyV2G:

    def __init__(self, threshold, actions_length, time_step_spec):
        self.threshold = tf.constant(threshold, tf.float32)
        self.actions_length = actions_length
        self._input_tensor_spec = time_step_spec.observation

    @property
    def input_tensor_spec(self):
        return self._input_tensor_spec

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        self._threshold = tf.constant(value, tf.float32)

    def action(self, timestep: TimeStep, policy_state, seed) -> PolicyStep:
        nest_utils.assert_matching_dtypes_and_inner_shapes(
            timestep.observation,
            self.input_tensor_spec,
            allow_extra_fields=True,
            caller=self,
            tensors_name="`timestep`",
            specs_name="`input_tensor_spec`")
        # observation = tf.squeeze(timestep.observation)
        observation = timestep.observation[0]
        current_price = observation[0:1]
        max_coefficient = observation[13:14]
        threshold_coefficient = observation[14:15]
        min_coefficient = observation[15:16]
        coefficient_step = (max_coefficient - min_coefficient) / (self.actions_length - 1)
        if coefficient_step == 0.0:
            return PolicyStep(action=tf.constant([self.actions_length - 1], dtype=tf.int64))
        good_price = current_price < self.threshold
        coefficient = threshold_coefficient
        if good_price:
            coefficient = (threshold_coefficient - max_coefficient) * (current_price / self.threshold) + max_coefficient

        # print(f", coefficient: {coefficient}")

        return PolicyStep(
            action=tf.cast(tf.round((coefficient - min_coefficient) / coefficient_step),
                                          dtype=tf.int64)
        )

