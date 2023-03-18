from typing import Callable, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from app.models.energy import EnergyCurve
from config import NUMBER_OF_AGENTS, DISCOUNT


class PowerMarketEnv(PyEnvironment):
#fix it, call to super class
    def __init__(self, energy_curve: EnergyCurve,
                 reward_function: Callable,
                 num_of_agents: int = NUMBER_OF_AGENTS,
                 train_mode: bool = True,):
        self._number_agents = num_of_agents
        self._reward_function = reward_function
        self._energy_curve = energy_curve
        self._mode = train_mode


        self._action_spec = array_spec.ArraySpec(
            shape=(), dtype=np.float32, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(13,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='observation'
        )

        self._time_of_day = 0
        self._day_ahead_prices = None
        self._intra_day_prices = None

        self._scaler = MinMaxScaler(copy=False)


    def batched(self) -> bool:
        return True

    def batch_size(self) -> Optional[int]:
        return self._number_agents

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def reward_spec(self) -> types.NestedArraySpec:
        return array_spec.ArraySpec(
            shape=(self._number_agents,),
            dtype=np.float32,
            name='reward'
        )

    def get_reward_function_name(self):
        return self._reward_function.__name__

    def _reset(self) -> ts.TimeStep:
        self._energy_curve.get_next_episode()
        self._time_of_day = 0
        self._day_ahead_prices = self._energy_curve.get_current_batch()
        self._intra_day_prices = self._energy_curve.get_current_batch_intra_day()
        obs = np.transpose(np.array(
            self._day_ahead_prices[:12] + self._intra_day_prices[self._time_of_day],
            ndmin=2,
            dtype=np.float32
        ))
        return ts.restart(
            observation=np.transpose(self._scaler.fit_transform(obs))[0],
            reward_spec=self.reward_spec(),
        )

    def _step(self, action: array_spec.ArraySpec):
        rewards_array = self._reward_function(self._day_ahead_prices,
                                            self._intra_day_prices[self._time_of_day],
                                            self._time_of_day,
                                            action,
                                            )
        #TODO add some kind of monitoring of prices
        self._time_of_day += 1
        self._time_of_day %= 24

        obs = np.array(
            [[i] for i in  self._day_ahead_prices[:12] + self._intra_day_prices[self._time_of_day]],
            dtype=np.float32,
        )

        if self._time_of_day > 0:
            return ts.transition(
                observation=self._scaler.fit_transform(obs).T[0],
                reward=np.array(rewards_array, dtype=np.float32),
                discount=DISCOUNT
            )
        else:
            return ts.termination(
                observation=self._scaler.fit_transform(obs).T[0],
                reward=np.array(rewards_array, dtype=np.float32),
            )



