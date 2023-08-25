from typing import Callable, List

import tensorflow as tf

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep

from app.models.tf_energy_3 import EnergyCurve
from config import DISCOUNT

class TFPowerMarketEnv(TFEnvironment, tf.Module):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 energy_curve: EnergyCurve,
                 reward_function: Callable,
                 number_of_agents: int,
                 avg_consumption_list: List[float],
                 train_mode: bool = True):
        super(TFPowerMarketEnv, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            batch_size=1,
        )
        self._number_of_agents = number_of_agents
        self._reward_function = reward_function
        self._energy_curve = energy_curve
        self._mode = train_mode
        self._hard_reset_flag = tf.Variable(False, dtype=tf.bool, trainable=False)
        self._discount = tf.constant(DISCOUNT, dtype=tf.float32)
        self._avg_consumption_tensor = tf.constant(avg_consumption_list, tf.float32)
        self._time_of_day = tf.Variable(-1, dtype=tf.int64)
        self._day_ahead_prices = tf.Variable([0.0] * 24 + [-1.0] * 12, dtype=tf.float32, trainable=False)
        self._intra_day_prices = tf.Variable([0.0] * 24 + [-1.0], dtype=tf.float32, trainable=False)
        self._last_time_step = tf.Variable([0.0] * (1 + 1 + number_of_agents + 13), dtype=tf.float32, trainable=False)
        self._day_ahead_prices_raw = tf.Variable([0.0] * 24, dtype=tf.float32, trainable=False)
        self._intra_day_prices_raw = tf.Variable([0.0] * 24, dtype=tf.float32, trainable=False)

    def get_reward_function_name(self):
        return self._reward_function.__name__

    def hard_reset(self):
        #print('Tracing hard_reset')
        # self._hard_reset_flag.assign(True)
        self._hard_reset_flag.assign(True)
        self._energy_curve.reset()

    # @tf.function
    def _return_last_timestep(self):
        #print('Tracing _return_last_timestep')
        tensor = self._last_time_step
        num_of_agents = self._number_of_agents
        step_type = tf.cast(tensor[0], tf.int64)
        discount = tensor[1]
        # reward = tf.reshape(tensor[2:num_of_agents+2], self.time_step_spec().reward.shape)
        reward = tensor[2:num_of_agents+2]
        observation = tensor[num_of_agents+2:]
        # observation = tensor[num_of_agents + 2:]
        return TimeStep(tf.expand_dims(step_type, axis=0),
                        tf.expand_dims(reward, axis=0),
                        tf.expand_dims(discount, axis=0),
                        tf.expand_dims(observation, axis=0))

    # @tf.function
    def _current_time_step(self):
        #print('Tracing _current_time_step')
        def a():
            rt = self._reset()
            self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                                   rt.discount,
                                                   tf.reshape(rt.reward, [-1]),
                                                   # rt.reward[0],
                                                   tf.squeeze(rt.observation)),
                                                  axis=0))
            return rt

        time = tf.cast(self._time_of_day, tf.int32)
        index = tf.where(tf.math.logical_or(tf.math.equal(24, time),
                                            tf.math.equal(-1, time)),
                         0,
                         1)
        rt = tf.switch_case(index, [a, self._return_last_timestep])
        return rt




    # @tf.function
    def _min_max_normalizer(self, tensor: tf.Tensor):
        #print('Tracing _min_max_normalizer')
        # if tf.squeeze(tensor).shape.rank > 1:
        #     raise Exception('Only normalizes rank <1 tensors')
        maximum = tf.reduce_max(tensor)
        temp = tf.where(tf.less(tensor, 0.0), maximum, tensor)
        minimum = tf.reduce_min(temp)
        # if minimum == maximum:
        #     if minimum == 0.0:
        #         return tensor - tensor
        #     else:
        #         return tensor - tensor + 0.5
        rt = tf.math.divide_no_nan(tensor - minimum, maximum - minimum)
        # possible_constant =
        # return (tensor - minimum) / (maximum - minimum)
        return tf.where(tf.less(rt, 0.0), -1.0, rt)
    # @tf.function
    def _get_obs(self, time):
        # print('Tracing _get_obs')
        da_prices = self._day_ahead_prices.value()[time:time+12]
        id_price = self._intra_day_prices.value()[time:time+1]
        # scaled_tensor = self._min_max_normalizer(tf.concat((da_prices, id_price), axis=0))
        # return tf.ensure_shape(scaled_tensor, [13])
        return tf.ensure_shape(tf.concat((da_prices, id_price), axis=0), [13])
    # @tf.function
    def _reset(self) -> ts.TimeStep:
        # print('Tracing _reset')
        # index = tf.where(self._hard_reset_flag, 2, tf.clip_by_value(tf.cast(self._time_of_day, tf.int32) - 23,
        #                                                             0,
        #                                                             1))
        # self._hard_reset_flag.assign(False)
        # tf.switch_case(index,
        #                [tf.no_op,
        #                 self._energy_curve.get_next_episode,
        #                 self._energy_curve.reset])
        self._time_of_day.assign(0)
        self._energy_curve.get_next_episode()
        da_prices = self._energy_curve.get_current_batch(normalized=True)
        self._day_ahead_prices.assign(tf.concat((da_prices[:24],
                                                 tf.constant([-1.0] * 12)), 0))
        da_prices_raw = self._energy_curve.get_current_batch(normalized=False)
        self._day_ahead_prices_raw.assign(da_prices_raw)
        id_batch = self._energy_curve.get_current_batch_intra_day(normalized=True)
        self._intra_day_prices.assign(tf.concat((id_batch[:24],
                                                 tf.constant([-1.0])), 0))
        id_batch_raw = self._energy_curve.get_current_batch_intra_day(normalized=False)
        self._intra_day_prices_raw.assign(id_batch_raw)
        obs = self._get_obs(0)
        rt = ts.restart(observation=tf.expand_dims(obs, 0),
                        batch_size=self.batch_size,
                        reward_spec=self.reward_spec())
        self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                               rt.discount,
                                               tf.reshape(rt.reward, [-1]),
                                               # rt.reward[0],
                                               tf.squeeze(rt.observation)),
                                              axis=0))
        return rt._replace(step_type=tf.cast(rt.step_type, tf.int64))

    # @tf.function
    def _step(self, action: tf.Tensor):
        # print('Tracing _step')
        def reward_tensor():
            reward_time = time  - 1
            return self._reward_function(self._day_ahead_prices_raw,
                                         self._intra_day_prices_raw[reward_time],
                                         reward_time,
                                         action,
                                         tf.gather(self._avg_consumption_tensor,
                                                   reward_time)) / (-10000.0)
        #TODO add some kind of monitoring of prices
        def a():
            rt = ts.transition(observation=tf.expand_dims(self._get_obs(time), 0),
                               reward=reward_tensor(),
                               discount=self._discount,
                               outer_dims=[self.batch_size])
            return rt._replace(step_type=tf.cast(rt.step_type, tf.int64))
        def b():
            rt = ts.termination(observation=tf.expand_dims(self._get_obs(time), 0),
                                reward=reward_tensor(),
                                outer_dims=[self.batch_size])
            return rt._replace(step_type=tf.cast(rt.step_type, tf.int64))
        self._time_of_day.assign_add(1)
        time = tf.cast(self._time_of_day.value(), tf.int32)
        index = tf.where(tf.math.logical_or(tf.math.equal(25, time),
                                            tf.math.equal(0, time)),
                         0,
                         tf.clip_by_value(time - 22, 1, 2)) #Trick to get 1 if time < 23, and 2 if time == 23
        rt = tf.switch_case(index, [self._reset, a, b])
        self._last_time_step.assign(tf.concat((tf.cast(rt.step_type, tf.float32),
                                               rt.discount,
                                               tf.reshape(rt.reward, [-1]),
                                               tf.squeeze(rt.observation)),
                                              axis=0))
        return rt