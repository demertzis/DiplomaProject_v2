import matplotlib.pyplot as plt
import numpy as np
from config import NUMBER_OF_AGENTS
from gym import spaces
from gym_anytrading.envs.trading_env import TradingEnv
from app.models.energy import EnergyCurve

import app.policies.reward_functions as rf

func_dict = {0: rf.vanilla,
             1: rf.halfway_uniform_rewards,
             2: rf.punishing_uniform_rewards,
             3: rf.punishing_non_uniform_rewards,
             4: rf.reinforcement_learning_rewards
             }


class PowerMarketEnv(TradingEnv):

	def __init__(self, energy_curve: EnergyCurve, reward_function):

		self.window_size = 0
		self.frame_bound = (0, 23)
		self.reward_function = func_dict[reward_function]
		self._episode_count = 0
		self._total_episodes = energy_curve.total_episodes()

		self.seed()
		self._energy_curve = energy_curve
		self.prices, self.signal_features = self._process_data()

		# episode
		self._step_reward = None
		self._per_agent_reward = None
		self._agent_identity = None
		self._done = None
		self._current_tick = None
		self._last_trade_tick = None
		self._position = None
		self._position_history = None
		self._total_step_reward = None
		self._total_reward = None
		self._total_profit = None
		self._first_rendering = None
		self._info = None
		self.history = None

		self._start_tick = 0
		self._end_tick = 24

		self.action_space = spaces.Box(low = -np.inf, high = np.inf, shape = (NUMBER_OF_AGENTS,), dtype = np.float32)
		self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (25,), dtype = np.float32)

	def reset(self):
		self._done = False
		self.prices, self.signal_features = self._process_data()
		self._episode_count = (self._episode_count + 1) % self._total_episodes
		if self._episode_count > 0:
			self._energy_curve.get_next_episode()
		self._per_agent_reward = np.full((NUMBER_OF_AGENTS,), 0., dtype = np.float32)
		self._step_reward = np.full((NUMBER_OF_AGENTS, 1), 0., dtype = np.float32)
		self._agent_identity = 0
		self._current_tick = self._start_tick
		self._total_reward = 0
		self._total_profit = 0.  # unit
		# self._first_rendering = True
		self.history = None
		return self._get_observation()

	def _get_observation(self):
		return np.insert(self.prices, 24, self.signal_features[self._current_tick])

	# return self.signal_features[(self._current_tick - self.window_size):self._current_tick + 1]

	def _process_data(self):
		# episode_frame = self._episode_count * 24
		# prices = self._energy_curve.loc[:, 'MCP_DAM'].to_numpy()[episode_frame: episode_frame + 24].astype(np.float32)
		prices = self._energy_curve.get_current_batch()

		# signal_features = self._energy_curve.loc[:, 'MCP_CRID'].to_numpy()[episode_frame: episode_frame + 24].astype(np.float32)
		signal_features = self._energy_curve.get_current_batch_intra_day()

		return prices, signal_features

	def step(self, action = None):
		self._done = False

		if (self._agent_identity == 0):
			self._step_reward = np.asarray(self._calculate_reward(action), dtype = np.float32)
			self._per_agent_reward += self._step_reward
			self._total_step_reward = np.sum(self._step_reward)
			self._update_profit(self._step_reward)

			self._info = dict(
				# agent_sum = self._step_reward,
				# total_agent_sum = self._per_agent_reward,
				total_reward = self._total_step_reward,
				total_profit = self._total_profit,
			)
			self._update_history(self._info)

		observation = self._get_observation()
		agent_reward = self._step_reward[self._agent_identity]
		self._agent_identity += 1

		if self._agent_identity == NUMBER_OF_AGENTS:
			self._current_tick += 1
			self._agent_identity = 0

		if self._current_tick == self._end_tick:
			self._done = True

		return observation, agent_reward, self._done, self._info

	def _calculate_reward(self, action):
		return self.reward_function(self.prices, self._get_observation()[24], self._current_tick, action)

	def _update_profit(self, step_reward):
		pass

	def render(self, mode = 'human'):
		def _plot_net_profit(tick):
			if not self.history:
				net_profit = 0.
			else:
				net_profit = self.history['total_reward'][tick]
			# if tick > 0:
			# 	net_profit -= self.history['total_reward'][tick - 1]
			if net_profit > 0.:
				color = 'red'
			else:
				color = 'green'
			if color:
				plt.scatter(tick, self.prices[tick], color = color)

		if self._first_rendering:
			self._first_rendering = False
			plt.cla()
			plt.plot(self.prices)
			_plot_net_profit(self._start_tick)

		_plot_net_profit(self._current_tick)

		plt.suptitle(
			"Total Reward: %.6f" % self._total_reward + ' ~ ' +
			"Total Profit: %.6f" % self._total_profit
		)

		plt.pause(0.01)

	def render_all(self, mode = 'human'):
		window_ticks = np.arange(len(self.history['total_reward']))
		plt.plot(self.prices)

		buy_ticks = []
		sell_ticks = []
		for i, tick in enumerate(window_ticks):
			if self.history['total_reward'][i] > 0:
				buy_ticks.append(tick)
			elif self.history['total_reward'][i] <= 0:
				sell_ticks.append(tick)

		plt.plot(buy_ticks, self.prices[buy_ticks], 'ro')
		plt.plot(sell_ticks, self.prices[sell_ticks], 'go')

		plt.suptitle(
			"Total Reward: %.6f" % self._total_reward + ' ~ ' +
			"Total Profit: %.6f" % self._total_profit
		)
