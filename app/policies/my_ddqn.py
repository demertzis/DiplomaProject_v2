from tf_agents.agents.dqn.dqn_agent import DdqnAgent
import tensorflow as tf
from tf_agents.utils import eager_utils


class MyDdqnAgent(DdqnAgent):
    def __init__(self, *args, **kwargs):
        super(MyDdqnAgent, self).__init__(*args, **kwargs)
        if not isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            raise Exception('Class should only be used with optimizers wrapped in LossScaleOptimizer'
                            '(Supposed to be used with float16 mixed_precision keras networks)')
    def _train(self, experience, weights):
        """
        Just added scaling of loss and unscaling of grads according
        to the custom training loop mixed precision tutorial of
        keras.mixed_precision
        """
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
            scaled_loss = self._optimizer.get_scaled_loss(loss_info.loss)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self._q_network.trainable_weights
        non_trainable_weights = self._q_network.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's q_network."
        scaled_grads = tape.gradient(scaled_loss, variables_to_train)
        grads = self._optimizer.get_unscaled_gradients(scaled_grads)
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = list(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            grads_and_vars_with_non_trainable = (
                    grads_and_vars + [(None, v) for v in non_trainable_weights])
            eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)
        self._optimizer.apply_gradients(grads_and_vars)
        self.train_step_counter.assign_add(1)

        self._update_target()

        return loss_info


