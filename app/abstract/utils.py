from tf_agents.environments.py_environment import PyEnvironment
import tensorflow as tf
from tf_agents.networks import network




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

class MyNetwork(network.Network):
    '''
    This is a custom wrapper for a non-stateful pre-defined tensorflow model
    '''
    def __init__(self,
                 model,
                 input_spec = None,
                 name = None):
        '''

        :param model: a tf.keras model, can be Functional or Sequential.
                      stateful network is currently not supported within this subclass;
                      i.e., the input model is expected to not have any `tf.keras.layers.{RNN,LSTM,GRU,...}` layer.
        :param input_spec: (Optional.) A nest of `tf.TypeSpec` representing the input observations to the first layer.
        :param name: (Optional.) Network name.
        '''
        super(MyNetwork, self).__init__(input_tensor_spec=input_spec,
                                         state_spec=(),
                                         name=name)
        self.model = model

    def copy(self, **kwargs) -> 'MyNetwork':
        """Make a copy of a `MyNetwork` instance.
        **NOTE** The new instance will not share weights with the original - but it will start with the same weights.
        Args:
          **kwargs: Args to override when recreating this network. Commonly overridden args include 'name'.
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
