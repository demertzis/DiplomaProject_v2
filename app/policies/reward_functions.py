import numpy as np

AVG_CONSUMPTION = 10.


def vanilla(prices: np.ndarray, crid_price: np.float32, tick: int, action: np.ndarray) -> np.ndarray:
	total_demand = np.sum(action)
	demand_overflow = AVG_CONSUMPTION - total_demand
	sell_price_min = crid_price * 0.8
	buy_price_max = prices[tick]
	if total_demand != 0:
		buy_price_max = (abs(min(0., demand_overflow)) * crid_price + min(AVG_CONSUMPTION, max(0., total_demand)) *
		                 prices[
			                 tick]) / total_demand
	else:
		return np.asarray([np.float32(0.0) for _ in action])

	neg_values = np.sum(action, where = [i < 0 for i in action])
	pos_values = np.sum(action, where = [i >= 0 for i in action])

	if total_demand >= 0:
		buy_price_final = ((pos_values - abs(neg_values)) * buy_price_max + abs(
			neg_values) * sell_price_min) / pos_values
		return np.asarray([np.float32(max(i, 0) * buy_price_final + min(i, 0) * sell_price_min) for i in action])

	if total_demand < 0:
		sell_price_final = ((abs(neg_values) - pos_values) * sell_price_min + pos_values * buy_price_max) / abs(
			neg_values)
		return np.asarray([np.float32(max(i, 0) * buy_price_max + min(i, 0) * sell_price_final) for i in action])


def halfway_uniform_rewards(prices: np.ndarray, crid_price: np.float32, tick: int, action: np.ndarray) -> np.ndarray:
	sell_price_min = crid_price * 0.8

	total_sell_load = abs(np.sum(action, where = [i < 0 for i in action]))
	total_buy_load = np.sum(action, where = [i >= 0 for i in action])

	common_load = min(total_sell_load, total_buy_load) if prices[tick] > sell_price_min else 0
	common_part = (sell_price_min + prices[tick]) / 2 * common_load if prices[tick] > sell_price_min else 0
	final_sell_price = ((total_sell_load - common_load) * sell_price_min + common_part) / total_sell_load
	final_buy_price = (max(0, total_buy_load - AVG_CONSUMPTION - common_load) * crid_price
	                   + common_part
	                   + min(AVG_CONSUMPTION, total_buy_load - common_load) * prices[tick]) / total_buy_load

	# rewards = np.asarray([i * (final_buy_price if i > 0 else final_sell_price) for i in action])
	# final_revenue = np.sum(rewards) - min(AVG_CONSUMPTION, np.sum(action)) * prices[tick] - max(0, np.sum(
	# 	action) - AVG_CONSUMPTION)
	# print(f'total rewards = {final_revenue}')
	return np.asarray([np.float32(i * (final_buy_price if i > 0 else final_sell_price)) for i in action])


def punishing_uniform_rewards(prices: np.ndarray, crid_price: np.float32, tick: int, action: np.ndarray):
	uniform_rewards = halfway_uniform_rewards(prices, crid_price, tick, action)
	total_sell_load = abs(np.sum(action, where = [i < 0 for i in action]))
	total_buy_load = np.sum(action, where = [i >= 0 for i in action])

	total_sell_amount = abs(np.sum(uniform_rewards, where = [i < 0 for i in uniform_rewards]))
	total_buy_amount = np.sum(uniform_rewards, where = [i >= 0 for i in uniform_rewards])
	day_avg_price = np.sum(prices) / len(prices)
	max_day_price = np.max(prices)
	min_day_price = np.min(prices)
	current_buy_price = total_buy_amount / total_buy_load
	current_sell_price = total_sell_amount / total_sell_load
	max_buy_price = (min(AVG_CONSUMPTION, total_buy_load) * prices[tick] +
	                 max(total_buy_load - AVG_CONSUMPTION, 0) * crid_price) / total_buy_load
	new_buy_price = None
	new_sell_price = None

	if max_buy_price > day_avg_price:
		total_max_buy_amount = max_buy_price * total_buy_load
		price_increase_ratio = np.exp(5 * (max_buy_price - day_avg_price) / (max_day_price - day_avg_price)) \
		                       / (np.exp((max_buy_price - day_avg_price) / (max_day_price - day_avg_price))
		                          + np.exp(5 * (1 - (max_buy_price - day_avg_price)) / (max_day_price - day_avg_price)))
		new_buy_price = current_buy_price + (max_buy_price - current_buy_price) * price_increase_ratio
		new_sell_price = current_sell_price + (new_buy_price - current_buy_price) * total_buy_load / total_sell_load


	elif max_buy_price <= day_avg_price:
		total_min_sell_amount = crid_price * 0.8 * total_buy_load
		price_decrease_ratio = np.exp((day_avg_price - max_buy_price) / (day_avg_price - min_day_price)) \
		                       / (np.exp((day_avg_price - max_buy_price) / (day_avg_price - min_day_price))
		                          + np.exp(5 * (1 - (day_avg_price - max_buy_price)) / (day_avg_price - min_day_price)))
		new_sell_price = current_sell_price - (current_sell_price - crid_price * 0.8) * price_decrease_ratio
		new_buy_price = current_buy_price - (current_sell_price - new_sell_price) * total_sell_load / total_buy_load

	print(f'punishing sum: {np.sum(np.asarray([i * (new_buy_price if i > 0 else new_sell_price) for i in action]))}\n')
	print(f'non-punishing sum: {np.sum(uniform_rewards)}\n')

	return np.asarray([np.float32(i * (new_buy_price if i > 0 else new_sell_price)) for i in action])


def compute_exp_weights(action: np.ndarray, CURRENT_BUY_PRICE: np.float32, CURRENT_SELL_PRICE: np.float32,
                        CRID_PRICE: np.float32,
                        MAX_VALUE: np.float32):
	total_buy_load = np.sum(action, where = [i >= 0 for i in action])
	total_sell_load = abs(np.sum(action, where = [i < 0 for i in action]))
	total_exp_buy_load = np.sum(np.asarray([np.exp( 5 * i / total_buy_load) for i in action if i >= 0]))
	total_exp_sell_load = np.sum(np.asarray([np.exp( 5 * abs(i) / total_sell_load) for i in action if i < 0]))
	weights_buy_list = []
	weights_sell_list = []

	for i, value in np.ndenumerate(action):
		if value >= 0:
			weights_buy_list.append((i[0], value))
		elif value < 0:
			weights_sell_list.append((i[0], abs(value)))

	weights_buy_list.sort(reverse = True, key = lambda x: x[1])
	weights_sell_list.sort(key = lambda x: x[1])

	running_total_exp_buy_load = total_exp_buy_load
	running_total_exp_sell_load = total_exp_sell_load
	running_total_buy_load = total_buy_load
	running_total_sell_load = total_sell_load
	running_total_buy_amount = total_buy_load * CURRENT_BUY_PRICE
	running_total_sell_amount = total_sell_load * CURRENT_SELL_PRICE

	buy_iter = iter(weights_buy_list)
	sell_iter = iter(weights_sell_list)
	final_list = np.asarray([0 for _ in range(len(action))], dtype = np.float32)

	while True:
		buy_element = next(buy_iter, False)

		if not buy_element:
			break
		else:
			weight = np.exp(5 * buy_element[1] / total_buy_load) / running_total_exp_buy_load
			overflow = weight * running_total_buy_amount > MAX_VALUE * buy_element[1]
			final_list[buy_element[0]] = MAX_VALUE if overflow else weight * running_total_buy_amount / buy_element[1]
			running_total_buy_amount -= final_list[buy_element[0]] * buy_element[1]
			running_total_buy_load -=  buy_element[1]
			running_total_exp_buy_load -= np.exp(5 * buy_element[1] / total_buy_load)

	while True:
		sell_element = next(sell_iter, False)

		if not sell_element:
			break
		else:
			weight = np.exp(5 * sell_element[1] / total_sell_load) / running_total_exp_sell_load
			overflow = weight * running_total_sell_amount < CRID_PRICE * sell_element[1]
			final_list[sell_element[0]] = CRID_PRICE if overflow else weight * running_total_sell_amount / sell_element[1]
			running_total_sell_amount -= final_list[sell_element[0]] * sell_element[1]
			running_total_sell_load -=  sell_element[1]
			running_total_exp_sell_load -= np.exp(5 * sell_element[1] / total_sell_load)

	return final_list

def punishing_non_uniform_rewards(prices: np.ndarray, crid_price: np.float32, tick: int, action: np.ndarray):
	punishing_rewards = halfway_uniform_rewards(prices, crid_price, tick, action)
	total_buy_load = np.sum(action, where = [i >= 0 for i in action])
	total_sell_load = abs(np.sum(action, where = [i < 0 for i in action]))
	UNIFORM_BUY_PRICE = np.sum(punishing_rewards, where = [i >= 0 for i in punishing_rewards]) / \
	                        total_buy_load
	UNIFORM_SELL_PRICE = abs(np.sum(punishing_rewards, where = [i < 0 for i in punishing_rewards])) / \
						    total_sell_load
	MAX_BUY_PRICE = (min(AVG_CONSUMPTION, total_buy_load) * prices[tick] +
	                 max(total_buy_load - AVG_CONSUMPTION, 0) * crid_price) / total_buy_load
	buy_sorted_arr = compute_exp_weights(action, UNIFORM_BUY_PRICE, UNIFORM_SELL_PRICE, crid_price * 0.8, MAX_BUY_PRICE)

	return np.asarray([np.float32(a * b) for a, b in zip(action, buy_sorted_arr)])

def reinforcement_learning_rewards(prices: np.ndarray, crid_price: np.float32, tick: int, action: np.ndarray):
	pass
