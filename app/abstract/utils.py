import logging

from tf_agents.environments.py_environment import PyEnvironment
import tensorflow as tf
from tf_agents.utils.common import Checkpointer



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



#Wasn't needed at the end (managed to pass pretrained network by layer extraction)
# class MyNetwork(network.Network):
#     '''
#     This is a custom wrapper for a non-stateful pre-defined tensorflow model
#     '''
#     def __init__(self,
#                  model,
#                  input_spec = None,
#                  _name = None):
#         '''
#
#         :param model: a tf.keras model, can be Functional or Sequential.
#                       stateful network is currently not supported within this subclass;
#                       i.e., the input model is expected to not have any `tf.keras.layers.{RNN,LSTM,GRU,...}` layer.
#         :param input_spec: (Optional.) A nest of `tf.TypeSpec` representing the input observations to the first layer.
#         :param _name: (Optional.) Network _name.
#         '''
#         super(MyNetwork, self).__init__(input_tensor_spec=input_spec,
#                                          state_spec=(),
#                                          _name=_name)
#         self.model = model
#
#     def copy(self, **kwargs) -> 'MyNetwork':
#         """Make a copy of a `MyNetwork` instance.
#         **NOTE** The new instance will not share weights with the original - but it will start with the same weights.
#         Args:
#           **kwargs: Args to override when recreating this network. Commonly overridden args include '_name'.
#         Returns:
#           A copy of this network.
#         Raises:
#           RuntimeError: If not `tf.executing_eagerly()`; as this is required to
#             be able to create deep copies of layers in `layers`.
#         """
#         new_kwargs = dict(self._saved_kwargs, **kwargs)
#         if 'model' not in kwargs:
#             new_model = tf.keras.models.clone_model(self.model)
#             new_kwargs['model'] = new_model
#         return type(self)(**new_kwargs)
#
#     def call(self, inputs, **kwargs):
#         # Only Networks are expected to know about step_type, network_state; not Keras models.
#         model_kwargs = kwargs.copy()
#         model_kwargs.pop('step_type', None)
#         model_kwargs.pop('network_state', None)
#         return self.model(inputs, **model_kwargs), ()
