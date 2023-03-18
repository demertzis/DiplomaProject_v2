from typing import Callable

import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep

from app.models.tf_energy import EnergyCurve
from config import DISCOUNT

class TFPowerMarketEnv(TFEnvironment):
    def __init__(self,
                 energy_curve: EnergyCurve,
                 reward_function: Callable,
                 number_of_agents: int,
                 train_mode: bool = True):
        time_step_spec = tensor_spec.from_spec(TimeStep(
            step_type=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=2),
            discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0.0, maximum=1.0),
            reward=tensor_spec.TensorSpec(shape=(number_of_agents,), dtype=tf.float32),
            observation=tensor_spec.BoundedTensorSpec(shape=(13,), dtype=tf.float32, minimum=-1., maximum=1.),
        ))
        action_spec = tensor_spec.from_spec(tensor_spec.TensorSpec(
            shape=(number_of_agents,), dtype=tf.float32, name="action"
        ))
        super(TFPowerMarketEnv, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            batch_size=1,
        )
        self._reward_function = reward_function
        self._energy_curve = energy_curve
        self._mode = train_mode
        self._discount = tf.constant(DISCOUNT, dtype=tf.float32)
        self._time_of_day = tf.Variable(-1, dtype=tf.int32)
        self._day_ahead_prices = tf.Variable([0.0] * 24, dtype=tf.float32)
        self._intra_day_prices = tf.Variable([0.0] * 24, dtype=tf.float32)
        self._last_time_step = tf.Variable([0.0] * (1 + 1 + number_of_agents + 13), dtype=tf.float32)

    def get_reward_function_name(self):
        return self._reward_function.__name__

    def _return_last_timestep(self):
        tensor = self._last_time_step
        num_of_agents = tf.shape(tensor)[0] - 15
        step_type = tf.cast(tensor[0], tf.int32)
        discount = tensor[1]
        reward = tf.reshape(tensor[2:num_of_agents+2], self.time_step_spec().reward.shape)
        observation = tensor[num_of_agents+2:]
        return TimeStep(tf.expand_dims(step_type, axis=0),
                        tf.expand_dims(reward, axis=0),
                        tf.expand_dims(discount, axis=0),
                        tf.expand_dims(observation, axis=0))

    def _current_time_step(self):
        if self._time_of_day == 24 or self._time_of_day == -1:
            rt = self._reset()
            self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                                   rt.discount,
                                                   tf.reshape(rt.reward, -1),
                                                   tf.squeeze(rt.observation)),
                                                  axis=0))
            return rt
        else:
            return self._return_last_timestep()

    def _min_max_normalizer(self, tensor: tf.Tensor):
        # if tf.squeeze(tensor).shape.rank > 1:
        #     raise Exception('Only normalizes rank <1 tensors')
        minimum = tf.reduce_min(tensor)
        maximum = tf.reduce_max(tensor)
        if minimum == maximum:
            if minimum == 0.0:
                return tensor - tensor
            else:
                return tensor - tensor + 0.5
        return (tensor - minimum) / (maximum - minimum)

    def _get_obs(self):
        time = self._time_of_day
        da_prices = self._day_ahead_prices[time:time+12]
        id_price = tf.cond(tf.less_equal(time, 23),
                           lambda: self._intra_day_prices[time],
                           lambda: tf.constant(0.0, dtype=tf.float32))
        id_price = tf.reshape(id_price, [1])
        scaled_tensor = self._min_max_normalizer(tf.concat((da_prices, id_price), axis=0))
        padding_length = 13 - tf.shape(scaled_tensor)[0]
        final_tensor = tf.pad(scaled_tensor, [[0, padding_length]], 'CONSTANT', -1.0)
        final_tensor = tf.ensure_shape(final_tensor, [13])
        return final_tensor

    def _reset(self) -> ts.TimeStep:
            if self._time_of_day == 24:
                self._energy_curve.get_next_episode()
            self._time_of_day.assign(0)
            self._day_ahead_prices.assign(self._energy_curve.get_current_batch())
            self._intra_day_prices.assign(self._energy_curve.get_current_batch_intra_day())

            obs = self._get_obs()

            rt = ts.restart(observation=tf.expand_dims(obs, 0),
                            batch_size=self.batch_size,
                            reward_spec=self.reward_spec(),
                           )
            self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                                   rt.discount,
                                                   tf.reshape(rt.reward, [-1]),
                                                   tf.squeeze(rt.observation)),
                                                  axis=0))
            return rt

    def _step(self, action: tf.Tensor):
        if self._time_of_day == 24 or self._time_of_day == -1:
            return self._reset()
        reward_tensor = self._reward_function(self._day_ahead_prices,
                                              self._intra_day_prices[self._time_of_day],
                                              self._time_of_day,
                                              action)
        #TODO add some kind of monitoring of prices
        self._time_of_day.assign_add(1)
        obs = self._get_obs()
        if self._time_of_day < 24:
            rt = ts.transition(
                observation=tf.expand_dims(obs, 0),
                reward=reward_tensor,
                discount=self._discount,
                outer_dims=[self.batch_size]#TODO check if it's right
            )
        else:
            #decide what to do. Probably reset and return a transition
            rt =  ts.termination(
                observation=tf.expand_dims(obs, 0),
                reward=reward_tensor,
                outer_dims=[self.batch_size]
            )
        self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                               rt.discount,
                                               tf.reshape(rt.reward, [-1]),
                                               tf.squeeze(rt.observation)),
                                              axis=0))
        return rt