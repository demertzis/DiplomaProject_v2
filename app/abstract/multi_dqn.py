from tf_agents.agents import tf_agent
from tf_agents.agents.dqn.dqn_agent import DdqnAgent, DqnLossInfo
import tensorflow as tf
from tf_agents.policies import q_policy, boltzmann_policy, epsilon_greedy_policy, greedy_policy
from tf_agents.networks import utils as network_utils
from tf_agents.utils import nest_utils, eager_utils
from tf_agents.utils import common

from app.abstract.utils import my_aggregate_losses
# class MultiDdqnAgent_2(DdqnAgent):
#     def _check_action_spec(self, action_spec):
#         flat_action_spec = tf.nest.flatten(action_spec)
#
#         # TODO(oars): Get DQN working with more than one dim in the actions.
#         if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 1:
#             raise ValueError(
#                 'Only scalar or 1-dimensional actions are supported now, but action spec is: {}'
#                 .format(action_spec))
#
#         spec = flat_action_spec[0]
#
#         # TODO(b/119321125): Disable this once index_with_actions supports
#         # negative-valued actions.
#         if spec.minimum != 0:
#             raise ValueError(
#                 'Action specs should have minimum of 0, but saw: {0}'.format(spec))
#
#         self._num_actions = spec.maximum - spec.minimum + 1
#
#     def _check_network_output(self, net, label):
#         """Check outputs of q_net and target_q_net against expected shape.
#
#         Subclasses that require different q_network outputs should override
#         this function.
#
#         Args:
#           net: A `Network`.
#           label: A label to print in case of a mismatch.
#         """
#         num_of_agents = self._q_network.layers[0].input_shape[1]
#         network_utils.check_single_floating_network_output(
#             net.create_variables(),
#             expected_output_shape=(num_of_agents ,self._num_actions,),
#             label=label)
#
#     def _setup_policy(self, time_step_spec, action_spec,
#                       boltzmann_temperature, emit_log_probability):
#
#         policy = q_policy.QPolicy(
#             time_step_spec,
#             action_spec,
#             q_network=self._q_network,
#             emit_log_probability=emit_log_probability,
#             observation_and_action_constraint_splitter=(
#                 self._observation_and_action_constraint_splitter),
#             validate_action_spec_and_network=False)
#
#         if boltzmann_temperature is not None:
#             collect_policy = boltzmann_policy.BoltzmannPolicy(
#                 policy, temperature=self._boltzmann_temperature)
#         else:
#             collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
#                 policy, epsilon=self._epsilon_greedy)
#         policy = greedy_policy.GreedyPolicy(policy)
#
#         # Create self._target_greedy_policy in order to compute target Q-values.
#         target_policy = q_policy.QPolicy(
#             time_step_spec,
#             action_spec,
#             q_network=self._target_q_network,
#             observation_and_action_constraint_splitter=(
#                 self._observation_and_action_constraint_splitter),
#             validate_action_spec_and_network=False)
#         self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)
#
#         return policy, collect_policy



class MultiDdqnAgent(DdqnAgent):

    def __init__(self, *args, **kwargs):
        num_of_agents = kwargs.pop("num_of_agents")
        if num_of_agents:
            self._grad_multiplier = tf.cast(num_of_agents, tf.float32)
        else:
            raise Exception('num_of_agents argument must be provided but it wasn\'t' )
        super(MultiDdqnAgent, self).__init__(*args, **kwargs)
    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)

        # TODO(oars): Get DQN working with more than one dim in the actions.
        if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 1:
            raise ValueError(
                'Only scalar or 1-dimensional actions are supported now, but action spec is: {}'
                .format(action_spec))

        spec = flat_action_spec[0]

        # TODO(b/119321125): Disable this once index_with_actions supports
        # negative-valued actions.
        if spec.minimum != 0:
            raise ValueError(
                'Action specs should have minimum of 0, but saw: {0}'.format(spec))

        self._num_actions = spec.maximum - spec.minimum + 1

    def _check_network_output(self, net, label):
        """Check outputs of q_net and target_q_net against expected shape.

        Subclasses that require different q_network outputs should override
        this function.

        Args:
          net: A `Network`.
          label: A label to print in case of a mismatch.
        """
        num_of_agents = self._q_network.layers[0].input_shape[1]
        network_utils.check_single_floating_network_output(
            net.create_variables(),
            expected_output_shape=(num_of_agents ,self._num_actions,),
            label=label)

    def _setup_policy(self, time_step_spec, action_spec,
                      boltzmann_temperature, emit_log_probability):

        policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._q_network,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter),
            validate_action_spec_and_network=False)

        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(
                policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = greedy_policy.GreedyPolicy(policy)

        # Create self._target_greedy_policy in order to compute target Q-values.
        target_policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._target_q_network,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter),
            validate_action_spec_and_network=False)
        self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)

        return policy, collect_policy

    # def _train(self, experience, weights):
    #     with tf.GradientTape() as tape:
    #         loss_info = self._loss(
    #             experience,
    #             td_errors_loss_fn=self._td_errors_loss_fn,
    #             gamma=self._gamma,
    #             reward_scale_factor=self._reward_scale_factor,
    #             weights=weights,
    #             training=True)
    #     tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    #     variables_to_train = self._q_network.trainable_weights
    #     non_trainable_weights = self._q_network.non_trainable_weights
    #     assert list(variables_to_train), "No variables in the agent's q_network."
    #     grads = tape.gradient(loss_info.loss, variables_to_train)
    #     # Tuple is used for py3, where zip is a generator producing values once.
    #     grads_and_vars = list(zip(grads, variables_to_train))
    #     raise Exception('Malakies')
    #     # if self._gradient_clipping is not None:
    #     #     grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
    #     #                                                      self._gradient_clipping)
    #
    #     # if self._summarize_grads_and_vars:
    #     #     grads_and_vars_with_non_trainable = (
    #     #             grads_and_vars + [(None, v) for v in non_trainable_weights])
    #     #     eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
    #     #                                         self.train_step_counter)
    #     #     eager_utils.add_gradients_summaries(grads_and_vars,
    #     #                                         self.train_step_counter)
    #     self._optimizer.apply_gradients(grads_and_vars)
    #     self.train_step_counter.assign_add(1)
    #
    #     self._update_target()
    #
    #     return loss_info

    # def _loss(self,
    #           experience,
    #           td_errors_loss_fn=None,
    #           gamma=1.0,
    #           reward_scale_factor=1.0,
    #           weights=None,
    #           training=False):
    #     """Computes loss for DQN training.
    #
    #     Args:
    #       experience: A batch of experience data in the form of a `Trajectory` or
    #         `Transition`. The structure of `experience` must match that of
    #         `self.collect_policy.step_spec`.
    #
    #         If a `Trajectory`, all tensors in `experience` must be shaped
    #         `[B, T, ...]` where `T` must be equal to `self.train_sequence_length`
    #         if that property is not `None`.
    #       td_errors_loss_fn: A function(td_targets, predictions) to compute the
    #         element wise loss.
    #       gamma: Discount for future rewards.
    #       reward_scale_factor: Multiplicative factor to scale rewards.
    #       weights: Optional scalar or elementwise (per-batch-entry) importance
    #         weights.  The output td_loss will be scaled by these weights, and
    #         the final scalar loss is the mean of these values.
    #       training: Whether this loss is being used for training.
    #
    #     Returns:
    #       loss: An instance of `DqnLossInfo`.
    #     Raises:
    #       ValueError:
    #         if the number of actions is greater than 1.
    #     """
    #     transition = self._as_transition(experience)
    #     time_steps, policy_steps, next_time_steps = transition
    #     actions = policy_steps.action
    #     # TODO(b/195943557) remove td_errors_loss_fn input to _loss
    #     self._td_errors_loss_fn = td_errors_loss_fn or self._td_errors_loss_fn
    #
    #     with tf.name_scope('loss'):
    #         q_values = self._compute_q_values(time_steps, actions, training=training)
    #
    #         next_q_values = self._compute_next_q_values(
    #             next_time_steps, policy_steps.info)
    #
    #         rewards = reward_scale_factor * next_time_steps.reward
    #         discounts = gamma * next_time_steps.discount
    #
    #         td_loss, td_error = self._td_loss(q_values, next_q_values, rewards,
    #                                           discounts)
    #
    #         valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
    #         td_error = valid_mask * td_error
    #
    #         td_loss = valid_mask * td_loss
    #
    #         if nest_utils.is_batched_nested_tensors(
    #                 time_steps, self.time_step_spec, num_outer_dims=2):
    #             # Do a sum over the time dimension.
    #             td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)
    #
    #         # Aggregate across the elements of the batch and add regularization loss.
    #         # Note: We use an element wise loss above to ensure each element is always
    #         #   weighted by 1/N where N is the batch size, even when some of the
    #         #   weights are zero due to boundary transitions. Weighting by 1/K where K
    #         #   is the actual number of non-zero weight would artificially increase
    #         #   their contribution in the loss. Think about what would happen as
    #         #   the number of boundary samples increases.
    #
    #         agg_loss = common.aggregate_losses(
    #             per_example_loss=td_loss,
    #             sample_weight=weights,
    #             regularization_loss=self._q_network.losses)
    #         total_loss = agg_loss.total_loss
    #
    #         losses_dict = {'td_loss': agg_loss.weighted,
    #                        'reg_loss': agg_loss.regularization,
    #                        'total_loss': total_loss}
    #
    #         common.summarize_scalar_dict(losses_dict,
    #                                      step=self.train_step_counter,
    #                                      name_scope='Losses/')
    #
    #         if self._summarize_grads_and_vars:
    #             with tf.name_scope('Variables/'):
    #                 for var in self._q_network.trainable_weights:
    #                     tf.compat.v2.summary.histogram(
    #                         name=var.name.replace(':', '_'),
    #                         data=var,
    #                         step=self.train_step_counter)
    #
    #         if self._debug_summaries:
    #             diff_q_values = q_values - next_q_values
    #             common.generate_tensor_summaries('td_error', td_error,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('td_loss', td_loss,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('q_values', q_values,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('next_q_values', next_q_values,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('diff_q_values', diff_q_values,
    #                                              self.train_step_counter)
    #
    #         return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
    #                                                          td_error=td_error))

    # def _loss(self,
    #           experience,
    #           td_errors_loss_fn=None,
    #           gamma=1.0,
    #           reward_scale_factor=1.0,
    #           weights=None,
    #           training=False):
    #     """Computes loss for DQN training.
    #
    #     Args:
    #       experience: A batch of experience data in the form of a `Trajectory` or
    #         `Transition`. The structure of `experience` must match that of
    #         `self.collect_policy.step_spec`.
    #
    #         If a `Trajectory`, all tensors in `experience` must be shaped
    #         `[B, T, ...]` where `T` must be equal to `self.train_sequence_length`
    #         if that property is not `None`.
    #       td_errors_loss_fn: A function(td_targets, predictions) to compute the
    #         element wise loss.
    #       gamma: Discount for future rewards.
    #       reward_scale_factor: Multiplicative factor to scale rewards.
    #       weights: Optional scalar or elementwise (per-batch-entry) importance
    #         weights.  The output td_loss will be scaled by these weights, and
    #         the final scalar loss is the mean of these values.
    #       training: Whether this loss is being used for training.
    #
    #     Returns:
    #       loss: An instance of `DqnLossInfo`.
    #     Raises:
    #       ValueError:
    #         if the number of actions is greater than 1.
    #     """
    #     transition = self._as_transition(experience)
    #     time_steps, policy_steps, next_time_steps = transition
    #     actions = policy_steps.action
    #     # TODO(b/195943557) remove td_errors_loss_fn input to _loss
    #     self._td_errors_loss_fn = td_errors_loss_fn or self._td_errors_loss_fn
    #
    #     with tf.name_scope('loss'):
    #         q_values = self._compute_q_values(time_steps, actions, training=training)
    #
    #         next_q_values = self._compute_next_q_values(
    #             next_time_steps, policy_steps.info)
    #
    #         rewards = reward_scale_factor * next_time_steps.reward
    #         discounts = gamma * next_time_steps.discount
    #
    #         td_loss, td_error = self._td_loss(q_values, next_q_values, rewards,
    #                                           discounts)
    #
    #         valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
    #         td_error = valid_mask * td_error
    #
    #         td_loss = valid_mask * td_loss
    #
    #         if nest_utils.is_batched_nested_tensors(
    #                 time_steps, self.time_step_spec, num_outer_dims=2):
    #             # Do a sum over the time dimension.
    #             td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)
    #
    #         # Aggregate across the elements of the batch and add regularization loss.
    #         # Note: We use an element wise loss above to ensure each element is always
    #         #   weighted by 1/N where N is the batch size, even when some of the
    #         #   weights are zero due to boundary transitions. Weighting by 1/K where K
    #         #   is the actual number of non-zero weight would artificially increase
    #         #   their contribution in the loss. Think about what would happen as
    #         #   the number of boundary samples increases.
    #
    #         agg_loss = my_aggregate_losses(
    #             per_example_loss=td_loss,
    #             sample_weight=weights,
    #             regularization_loss=self._q_network.losses)
    #         total_loss = agg_loss.total_loss
    #         #Used to expect scalar losses. Now because losses have a single dimmension we just get the mean of all agents
    #         losses_dict = {'td_loss': tf.reduce_mean(agg_loss.weighted),
    #                        'reg_loss': tf.reduce_mean(agg_loss.regularization),
    #                        'total_loss': tf.reduce_mean(total_loss)}
    #
    #         common.summarize_scalar_dict(losses_dict,
    #                                      step=self.train_step_counter,
    #                                      name_scope='Losses/')
    #
    #         if self._summarize_grads_and_vars:
    #             with tf.name_scope('Variables/'):
    #                 for var in self._q_network.trainable_weights:
    #                     tf.compat.v2.summary.histogram(
    #                         name=var.name.replace(':', '_'),
    #                         data=var,
    #                         step=self.train_step_counter)
    #
    #         if self._debug_summaries:
    #             diff_q_values = q_values - next_q_values
    #             common.generate_tensor_summaries('td_error', td_error,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('td_loss', td_loss,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('q_values', q_values,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('next_q_values', next_q_values,
    #                                              self.train_step_counter)
    #             common.generate_tensor_summaries('diff_q_values', diff_q_values,
    #                                              self.train_step_counter)
    #
    #         return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
    #                                                          td_error=td_error))


    def _train(self, experience, weights):
        """
        Rescaled the loss by multiplying by the number of agents. This is done because the loss function considers the
        agents axis of the reward tensor as a time axis and reduces the tensor to a mean. The loss that's returned is
        the mean without the scaling used for gradient calculation.
        """
        experience = experience.replace(step_type=tf.expand_dims(experience.step_type, axis=-1),
                                        next_step_type=tf.expand_dims(experience.next_step_type, axis=-1),
                                        discount=tf.expand_dims(experience.discount, axis=-1),)
        with tf.GradientTape() as tape:
          loss_info = self._loss(
              experience,
              td_errors_loss_fn=self._td_errors_loss_fn,
              gamma=self._gamma,
              reward_scale_factor=self._reward_scale_factor,
              weights=weights,
              training=True)
          scaled_loss = loss_info.loss * self._grad_multiplier
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self._q_network.trainable_weights
        non_trainable_weights = self._q_network.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's q_network."
        grads = tape.gradient(scaled_loss, variables_to_train)
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

    def _loss(self,
              experience,
              td_errors_loss_fn=None,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None,
              training=False):
        """Computes loss for DQN training.

        !!!Just made a change where the tensors are checked for having a time dimension. The original version compares the
        shape of experience with self.time_step_spec which we have chaged to the shape of expected experience via
        the training_data_spec argument (added 1 dimension to step_type and discount in order to allow other tf operations
        to work, nothing major)!!!s

        Args:
          experience: A batch of experience data in the form of a `Trajectory` or
            `Transition`. The structure of `experience` must match that of
            `self.collect_policy.step_spec`.

            If a `Trajectory`, all tensors in `experience` must be shaped
            `[B, T, ...]` where `T` must be equal to `self.train_sequence_length`
            if that property is not `None`.
          td_errors_loss_fn: A function(td_targets, predictions) to compute the
            element wise loss.
          gamma: Discount for future rewards.
          reward_scale_factor: Multiplicative factor to scale rewards.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.  The output td_loss will be scaled by these weights, and
            the final scalar loss is the mean of these values.
          training: Whether this loss is being used for training.

        Returns:
          loss: An instance of `DqnLossInfo`.
        Raises:
          ValueError:
            if the number of actions is greater than 1.
        """
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action
        # TODO(b/195943557) remove td_errors_loss_fn input to _loss
        self._td_errors_loss_fn = td_errors_loss_fn or self._td_errors_loss_fn

        with tf.name_scope('loss'):
            q_values = self._compute_q_values(time_steps, actions, training=training)

            next_q_values = self._compute_next_q_values(
                next_time_steps, policy_steps.info)

            rewards = reward_scale_factor * next_time_steps.reward
            discounts = gamma * next_time_steps.discount

            td_loss, td_error = self._td_loss(q_values, next_q_values, rewards,
                                              discounts)

            valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
            td_error = valid_mask * td_error

            td_loss = valid_mask * td_loss

            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.data_context.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            # Note: We use an element wise loss above to ensure each element is always
            #   weighted by 1/N where N is the batch size, even when some of the
            #   weights are zero due to boundary transitions. Weighting by 1/K where K
            #   is the actual number of non-zero weight would artificially increase
            #   their contribution in the loss. Think about what would happen as
            #   the number of boundary samples increases.

            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights,
                regularization_loss=self._q_network.losses)
            total_loss = agg_loss.total_loss

            losses_dict = {'td_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self._q_network.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common.generate_tensor_summaries('td_error', td_error,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('next_q_values', next_q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                 self.train_step_counter)

            return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
                                                             td_error=td_error))

