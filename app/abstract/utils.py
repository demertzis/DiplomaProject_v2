import logging
from typing import Sequence, Callable, Any, Optional, Tuple

import numpy as np
from tf_agents.drivers import driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import tf_environment
from tf_agents.environments.py_environment import PyEnvironment
import tensorflow as tf
from tf_agents.networks import network
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.losses import util as losses_util
from tf_agents.policies import tf_policy
from tf_agents.trajectories import Trajectory, Transition, policy_step, trajectory
from tf_agents.trajectories.trajectory import _validate_rank
from tf_agents.typing import types
from tf_agents.utils import composite, common
import tf_agents.trajectories.time_step as ts
from tf_agents.utils.common import Checkpointer, AggregatedLosses


class MyCheckpointer(Checkpointer):
    def __init__(self, ckpt_dir, max_to_keep=20, **kwargs):
        """A class for making checkpoints.

        If ckpt_dir doesn't exist it creates it.

        Args:
          ckpt_dir: The directory to save checkpoints.
          max_to_keep: Maximum number of checkpoints to keep (if greater than the
            max are saved, the oldest checkpoints are deleted).
          **kwargs: Items to include in the checkpoint.
        """
        self._checkpoint = tf.train.Checkpoint(**kwargs)

        if not tf.io.gfile.exists(ckpt_dir):
            tf.io.gfile.makedirs(ckpt_dir)

        self._manager = tf.train.CheckpointManager(
            self._checkpoint, directory=ckpt_dir, max_to_keep=max_to_keep)

        if self._manager.latest_checkpoint is not None:
            logging.info('Checkpoint available: %s', self._manager.latest_checkpoint)
            self._checkpoint_exists = True
        else:
            logging.info('No checkpoint available at %s', ckpt_dir)
            self._checkpoint_exists = False
        self._load_status = None
        # self._load_status = self._checkpoint.restore(
        #     self._manager.latest_checkpoint)

    def initialize_or_restore(self, session=None):
        """
        Modified function to be able to create checkpointer without actually loading the checkpoint on initialization.
        In effect, you need to create the checkpointer (class MyCheckpointer) and call intialize or restore whenever
        you want to load checkpointer
        """
        self._load_status = self._checkpoint.restore(
                self._manager.latest_checkpoint)
        self._load_status.initialize_or_restore(session)
        return self._load_status


def compute_avg_return(environment: PyEnvironment, policy, num_episodes=10):
    total_return = 0.0
    step = 0

    for _ in range(num_episodes):

        # Reset the environment
        time_step = environment.reset()
        # Initialize the episode return
        episode_return = 0.0

        # While the current state is not a terminal state
        while not time_step.is_last():
            # Use policy to get the next action
            action_step = policy.action(time_step)
            # Apply action on the environment to get next state
            time_step = environment.step(action_step.action)
            # Add reward on the episode return
            episode_return += time_step.reward
            # Increase step counter
            step += 1
        # Add episode return on total_return counter
        total_return += episode_return

    # Calculate average return
    avg_return = total_return / step
    # Unpack value
    return avg_return.numpy()[0]

# class MyCheckpointer(Checkpointer):
#     def change_dir(self, new_dir):
#         if not tf.io.gfile.exists(new_dir):
#             tf.io.gfile.makedirs(new_dir)



# Wasn't needed at the end (managed to pass pretrained network by layer extraction)
class MyNetwork(network.Network):
    '''
    This is a custom wrapper for a non-stateful pre-defined tensorflow model
    '''
    def __init__(self,
                 model,
                 input_spec = None,
                 _name = None):
        '''

        :param model: a tf.keras model, can be Functional or Sequential.
                      stateful network is currently not supported within this subclass;
                      i.e., the input model is expected to not have any `tf.keras.layers.{RNN,LSTM,GRU,...}` layer.
        :param input_spec: (Optional.) A nest of `tf.TypeSpec` representing the input observations to the first layer.
        :param _name: (Optional.) Network _name.
        '''
        super(MyNetwork, self).__init__(input_tensor_spec=input_spec,
                                         state_spec=(),
                                         name=_name)
        self.model = model

    def copy(self, **kwargs) -> 'MyNetwork':
        """Make a copy of a `MyNetwork` instance.
        **NOTE** The new instance will not share weights with the original - but it will start with the same weights.
        Args:
          **kwargs: Args to override when recreating this network. Commonly overridden args include '_name'.
        Returns:
          A copy of this network.
        Raises:
          RuntimeError: If not `tf.executing_eagerly()`; as this is required to
            be able to create deep copies of layers in `layers`.
        """
        new_kwargs = dict(self._saved_kwargs, **kwargs)
        if 'model' not in kwargs:
            new_model = tf.keras.models.clone_model(self.model)
            new_kwargs['model'] = new_model
        return type(self)(**new_kwargs)

    def call(self, inputs, **kwargs):
        # Only Networks are expected to know about step_type, network_state; not Keras models.
        model_kwargs = kwargs.copy()
        model_kwargs.pop('step_type', None)
        model_kwargs.pop('network_state', None)
        return self.model(inputs, **model_kwargs), ()

class MyDynamicEpisodeDriver(DynamicEpisodeDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_fn = tf.function(jit_compile=True)(self._run)

def my_discounted_return(rewards,
                         discounts,
                         final_value=None,
                         time_major=True,
                         provide_all_returns=True):
    """Computes discounted return.

    ```
    Q_n = sum_{n'=n}^N gamma^(n'-n) * r_{n'} + gamma^(N-n+1)*final_value.
    ```

    For details, see
    "Reinforcement Learning: An Introduction" Second Edition
    by Richard S. Sutton and Andrew G. Barto

    Define abbreviations:
    `B`: batch size representing number of trajectories.
    `T`: number of steps per trajectory.  This is equal to `N - n` in the equation
         above.

    **Note** To replicate the calculation `Q_n` exactly, use
    `discounts = gamma * tf.ones_like(rewards)` and `provide_all_returns=False`.

    Args:
      rewards: Tensor with shape `[T, B]` (or `[T]`) representing rewards.
      discounts: Tensor with shape `[T, B]` (or `[T]`) representing discounts.
      final_value: (Optional.).  Default: An all zeros tensor.  Tensor with shape
        `[B]` (or `[1]`) representing value estimate at `T`. This is optional;
        when set, it allows final value to bootstrap the reward computation.
      time_major: A boolean indicating whether input tensors are time major. False
        means input tensors have shape `[B, T]`.
      provide_all_returns: A boolean; if True, this will provide all of the
        returns by time dimension; if False, this will only give the single
        complete discounted return.

    Returns:
      If `provide_all_returns`:
        A tensor with shape `[T, B]` (or `[T]`) representing the discounted
        returns. The shape is `[B, T]` when `not time_major`.
      If `not provide_all_returns`:
        A tensor with shape `[B]` (or []) representing the discounted returns.
    """
    if not time_major:
      with tf.name_scope("to_time_major_tensors"):
          discounts = tf.transpose(discounts, perm=[1, 0, 2])
          # discounts = tf.transpose(discounts)
          # rewards = tf.transpose(rewards, perm=[1, 0] + list(range(discounts.shape.rank - 2)))
          rewards = tf.transpose(rewards, perm=[1, 0, 2])

    if final_value is None:
        final_value = tf.zeros_like(rewards[-1])

    def discounted_return_fn(accumulated_discounted_reward, reward_discount):
        reward, discount = reward_discount
        return accumulated_discounted_reward * discount + reward

    if provide_all_returns:
      returns = tf.nest.map_structure(
          tf.stop_gradient,
          tf.scan(
              fn=discounted_return_fn,
              elems=(rewards, discounts),
              reverse=True,
              initializer=final_value))

      if not time_major:
        with tf.name_scope("to_batch_major_tensors"):
          returns = tf.transpose(rewards, perm=[1, 0, 2])
    else:
      returns = tf.foldr(
          fn=discounted_return_fn,
          elems=(rewards, discounts),
          initializer=final_value,
          back_prop=False),

    return tf.stop_gradient(returns[0])

def my_index_with_actions(q_values, actions, multi_dim_actions=False):
    """Index into q_values using actions.

    Note: this supports multiple outer dimensions (e.g. time, batch etc).

    Args:
      q_values: A float tensor of shape [outer_dim1, ... outer_dimK, action_dim1,
        ..., action_dimJ].
      actions: An int tensor of shape [outer_dim1, ... outer_dimK]    if
        multi_dim_actions=False [outer_dim1, ... outer_dimK, J] if
        multi_dim_actions=True I.e. in the multidimensional case,
        actions[outer_dim1, ... outer_dimK] is a vector [actions_1, ...,
        actions_J] where each element actions_j is an action in the range [0,
        num_actions_j). While in the single dimensional case, actions[outer_dim1,
        ... outer_dimK] is a scalar.
      multi_dim_actions: whether the actions are multidimensional.

    Returns:
      A [outer_dim1, ... outer_dimK] tensor of q_values for the given actions.

    Raises:
      ValueError: If actions have unknown rank.
    """
    actions = tf.expand_dims(actions, axis=-1)
    if actions.shape.rank is None:
          raise ValueError('actions should have known rank.')
    batch_dims = actions.shape.rank
    if multi_dim_actions:
          # In the multidimensional case, the last dimension of actions indexes the
          # vector of actions for each batch, so exclude it from the batch dimensions.
          batch_dims -= 1

    outer_shape = tf.shape(input=actions)
    batch_indices = tf.meshgrid(
            *[tf.range(outer_shape[i]) for i in range(batch_dims)], indexing='ij')
    batch_indices = [tf.cast(tf.expand_dims(batch_index, -1), dtype=tf.int32)
                      for batch_index in batch_indices]
    if not multi_dim_actions:
          actions = tf.expand_dims(actions, -1)
    # Cast actions to tf.int32 in order to avoid a TypeError in tf.concat.
    actions = tf.cast(actions, dtype=tf.int32)
    action_indices = tf.concat(batch_indices + [actions], -1)
    return tf.gather_nd(q_values, action_indices)

def my_aggregate_losses(per_example_loss=None,
                        sample_weight=None,
                        regularization_loss=None):
    """Aggregates and scales per example loss and regularization losses.

    !!!Changed some parameters in order to not sum up the last dimension of losses to allow our multi agent framework!!!

    If `global_batch_size` is given it would be used for scaling, otherwise it
    would use the batch_dim of per_example_loss and number of replicas.

    Args:
      per_example_loss: Per-example loss [B] or [B, T, ...].
      sample_weight: Optional weighting for each example, Tensor shaped [B] or
        [B, T, ...], or a scalar float.
      global_batch_size: Optional global batch size value. Defaults to (size of
      first dimension of `losses`) * (number of replicas).
      regularization_loss: Regularization loss.

    Returns:
      An AggregatedLosses named tuple with scalar losses to optimize.
    """
    total_loss, weighted_loss, reg_loss = None, None, None
    if sample_weight is not None and not isinstance(sample_weight, tf.Tensor):
      sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32)

    # Compute loss that is scaled by global batch size.
    if per_example_loss is not None:
        loss_rank = per_example_loss.shape.rank
        if sample_weight is not None:
              weight_rank = sample_weight.shape.rank
              # Expand `sample_weight` to be broadcastable to the shape of
              # `per_example_loss`, to ensure that multiplication works properly.
              if weight_rank > 0 and loss_rank > weight_rank:
                for dim in range(weight_rank, loss_rank):
                  sample_weight = tf.expand_dims(sample_weight, dim)
              # Sometimes we have an episode boundary or similar, and at this location
              # the loss is nonsensical (i.e., inf or nan); and sample_weight is zero.
              # In this case, we should respect the zero sample_weight and ignore the
              # frame.
              per_example_loss = tf.math.multiply_no_nan(
                  per_example_loss, sample_weight)

        if loss_rank is not None and loss_rank == 0:
            err_msg = (
                'Need to use a loss function that computes losses per sample, ex: '
                'replace losses.mean_squared_error with tf.math.squared_difference. '
                'Invalid value passed for `per_example_loss`. Expected a tensor '
                'tensor with at least rank 1, received: {}'.format(per_example_loss))
            if tf.distribute.has_strategy():
                  raise ValueError(err_msg)
            else:
                  logging.warning(err_msg)
                  # Add extra dimension to prevent error in compute_average_loss.
                  per_example_loss = tf.expand_dims(per_example_loss, 0)
        elif loss_rank > 2:
            # If per_example_loss is shaped [B, T, ...], we need to compute the mean
            # across the extra dimensions, ex. time, as well.
            per_example_loss = tf.reduce_mean(per_example_loss, range(1, loss_rank))

        weighted_loss = my_compute_average_loss(
            per_example_loss,)
        total_loss = weighted_loss
    # Add scaled regularization losses.
    if regularization_loss is not None:
        reg_loss = tf.nn.scale_regularization_loss(regularization_loss)
        if total_loss is None:
          total_loss = reg_loss
        else:
          total_loss += reg_loss
    return AggregatedLosses(total_loss, weighted_loss, reg_loss)

def my_compute_average_loss(per_example_loss,
                            sample_weight=None):
  """Scales per-example losses with sample_weights and computes their average.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(labels, predictions, sample_weight=None):

      # If you are using a `Loss` class instead, set reduction to `NONE` so that
      # we can do the reduction afterwards and divide by global batch size.
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)
  ```

  Args:
    per_example_loss: Per-example loss.
    sample_weight: Optional weighting for each example.

  Returns:
    Scalar loss value.
  """  # pylint: disable=g-doc-exception
  per_example_loss = ops.convert_to_tensor(per_example_loss)
  input_dtype = per_example_loss.dtype

  with losses_util.check_per_example_loss_rank(per_example_loss):
    if sample_weight is not None:
      sample_weight = ops.convert_to_tensor(sample_weight)
      per_example_loss = losses_util.scale_losses_by_sample_weight(
          per_example_loss, sample_weight)
      per_example_loss = math_ops.cast(per_example_loss, input_dtype)
    num_replicas = ds.get_strategy().num_replicas_in_sync
    per_replica_batch_size = array_ops.shape_v2(per_example_loss)[0]
    global_batch_size = per_replica_batch_size * num_replicas

    global_batch_size = math_ops.cast(global_batch_size, input_dtype)
    return math_ops.reduce_sum(per_example_loss, axis=0) / global_batch_size


def my_to_n_step_transition(
    trajectory: Trajectory,
    gamma: types.Float
) -> Transition:
  """Create an n-step transition from a trajectory with `T=N + 1` frames.

  **NOTE** Tensors of `trajectory` are sliced along their *second* (`time`)
  dimension, to pull out the appropriate fields for the n-step transitions.

  The output transition's `next_time_step.{reward, discount}` will contain
  N-step discounted reward and discount values calculated as:

  ```
  next_time_step.reward = r_t +
                          g^{1} * d_t * r_{t+1} +
                          g^{2} * d_t * d_{t+1} * r_{t+2} +
                          g^{3} * d_t * d_{t+1} * d_{t+2} * r_{t+3} +
                          ...
                          g^{N-1} * d_t * ... * d_{t+N-2} * r_{t+N-1}
  next_time_step.discount = g^{N-1} * d_t * d_{t+1} * ... * d_{t+N-1}
  ```

  In python notation:

  ```python
  discount = gamma**(N-1) * reduce_prod(trajectory.discount[:, :-1])
  reward = discounted_return(
      rewards=trajectory.reward[:, :-1],
      discounts=gamma * trajectory.discount[:, :-1])
  ```

  When `trajectory.discount[:, :-1]` is an all-ones tensor, this is equivalent
  to:

  ```python
  next_time_step.discount = (
      gamma**(N-1) * tf.ones_like(trajectory.discount[:, 0]))
  next_time_step.reward = (
      sum_{n=0}^{N-1} gamma**n * trajectory.reward[:, n])
  ```

  Args:
    trajectory: An instance of `Trajectory`. The tensors in Trajectory must have
      shape `[B, T, ...]`.  `discount` is assumed to be a scalar float,
      hence the shape of `trajectory.discount` must be `[B, T]`.
    gamma: A floating point scalar; the discount factor.

  Returns:
    An N-step `Transition` where `N = T - 1`.  The reward and discount in
    `time_step.{reward, discount}` are NaN.  The n-step discounted reward
    and final discount are stored in `next_time_step.{reward, discount}`.
    All tensors in the `Transition` have shape `[B, ...]` (no time dimension).

  Raises:
    ValueError: if `discount.shape.rank != 2`.
    ValueError: if `discount.shape[1] < 2`.
  """
  _validate_rank(trajectory.discount, min_rank=3, max_rank=3)

  # Use static values when available, so that we can use XLA when the time
  # dimension is fixed.
  time_dim = (tf.compat.dimension_value(trajectory.discount.shape[1])
              or tf.shape(trajectory.discount)[1])

  static_time_dim = tf.get_static_value(time_dim)
  if static_time_dim in (0, 1):
    raise ValueError(
        'Trajectory frame count must be at least 2, but saw {}.  Shape of '
        'trajectory.discount: {}'.format(static_time_dim,
                                         trajectory.discount.shape))

  n = time_dim - 1

  # Use composite calculations to ensure we properly handle SparseTensor etc in
  # the observations.

  # pylint: disable=g-long-lambda

  # Pull out x[:,0] for x in trajectory
  first_frame = tf.nest.map_structure(
      lambda t: composite.squeeze(
          composite.slice_to(t, axis=1, end=1),
          axis=1),
      trajectory)

  # Pull out x[:,-1] for x in trajectory
  final_frame = tf.nest.map_structure(
      lambda t: composite.squeeze(
          composite.slice_from(t, axis=1, start=-1),
          axis=1),
      trajectory)
  # pylint: enable=g-long-lambda

  # When computing discounted return, we need to throw out the last time
  # index of both reward and discount, which are filled with dummy values
  # to match the dimensions of the observation.
  reward = trajectory.reward[:, :-1]
  discount = trajectory.discount[:, :-1]

  policy_steps = policy_step.PolicyStep(
      action=first_frame.action, state=(), info=first_frame.policy_info)

  discounted_reward = my_discounted_return(
      rewards=reward,
      discounts=gamma * discount,
      time_major=False,
      provide_all_returns=False)

  # NOTE: `final_discount` will have one less discount than `discount`.
  # This is so that when the learner/update uses an additional
  # discount (e.g. gamma) we don't apply it twice.
  final_discount = gamma**(n-1) * tf.math.reduce_prod(discount, axis=1)

  time_steps = ts.TimeStep(
      first_frame.step_type,
      # unknown
      reward=tf.nest.map_structure(
          lambda r: np.nan * tf.ones_like(r), first_frame.reward),
      # unknown
      discount=np.nan * tf.ones_like(first_frame.discount),
      observation=first_frame.observation)
  next_time_steps = ts.TimeStep(
      step_type=final_frame.step_type,
      reward=discounted_reward,
      discount=final_discount,
      observation=final_frame.observation)
  return Transition(time_steps, policy_steps, next_time_steps)




class MyTFDriver(driver.Driver):
  """A driver that runs a TF policy in a TF environment."""

  def __init__(
      self,
      env: tf_environment.TFEnvironment,
      policy: tf_policy.TFPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None,
      disable_tf_function: bool = False):
    """A driver that runs a TF policy in a TF environment.

    **Note** about bias when using batched environments with `max_episodes`:
    When using `max_episodes != None`, a `run` step "finishes" when
    `max_episodes` have been completely collected (hit a boundary).
    When used in conjunction with environments that have variable-length
    episodes, this skews the distribution of collected episodes' lengths:
    short episodes are seen more frequently than long ones.
    As a result, running an `env` of `N > 1` batched environments
    with `max_episodes >= 1` is not the same as running an env with `1`
    environment with `max_episodes >= 1`.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.TFPolicy policy.
      observers: A list of observers that are notified after every step
        in the environment. Each observer is a callable(trajectory.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)). The transition is shaped just as
        trajectories are for regular observers.
      max_steps: Optional maximum number of steps for each run() call. For
        batched or parallel environments, this is the maximum total number of
        steps summed across all environments. Also see below.  Default: 0.
      max_episodes: Optional maximum number of episodes for each run() call. For
        batched or parallel environments, this is the maximum total number of
        episodes summed across all environments. At least one of max_steps or
        max_episodes must be provided. If both are set, run() terminates when at
        least one of the conditions is
        satisfied.  Default: 0.
      disable_tf_function: If True the use of tf.function for the run method is
        disabled.

    Raises:
      ValueError: If both max_steps and max_episodes are None.
    """
    common.check_tf1_allowed()
    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    super(MyTFDriver, self).__init__(env, policy, observers, transition_observers)

    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf

    if not disable_tf_function:
      self.run = common.function(self.run, autograph=True)

  def run(self,
          time_step: ts.TimeStep,
          policy_state: types.NestedTensor = ()) -> Tuple[ts.TimeStep, types.NestedTensor]:
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = tf.constant(0.0)
    num_episodes = tf.constant(0.0)

    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step =  self.env.step(action_step.action)
      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += tf.math.reduce_sum(
          tf.cast(traj.is_boundary(), tf.float32))
      num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state






