import tensorflow as tf

from config import CRID_COEFFICIENT


def get_buy_price(total_demand, avg_consumption, da_prices, crid_price):
    abs_demand = tf.math.abs(total_demand)
    demand_overflow = total_demand - avg_consumption
    buy_price = tf.clip_by_value(tf.math.divide_no_nan(demand_overflow, abs_demand), 0.0, 1.0) * crid_price + \
                tf.clip_by_value(tf.math.divide_no_nan(tf.minimum(avg_consumption, total_demand), abs_demand), 0.0,
                                 1.0) * da_prices


def _get_basic_values(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor,
                      avg_consumption: tf.Tensor):
    total_demand = tf.math.reduce_sum(action)
    abs_demand = tf.math.abs(total_demand)
    demand_overflow = total_demand - avg_consumption
    buy_price = tf.clip_by_value(tf.math.divide_no_nan(demand_overflow, abs_demand), 0.0, 1.0) * crid_price + \
                tf.clip_by_value(tf.math.divide_no_nan(tf.minimum(avg_consumption, total_demand), abs_demand), 0.0,
                                 1.0) * prices[tick]
    sell_price = crid_price * CRID_COEFFICIENT * tf.cast(tf.less(total_demand, 0.0), tf.float32)
    return total_demand, buy_price, sell_price, crid_price * CRID_COEFFICIENT, tf.maximum(prices[tick], crid_price)


# def vanilla(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor, avg_consumption: tf.Tensor) -> tf.Tensor:
# 	total_demand = tf.math.reduce_sum(action)
# 	demand_overflow = total_demand - avg_consumption
# 	sell_price_min = crid_price * CRID_COEFFICIENT
# 	total_sell = total_demand * sell_price_min
# 	total_buy = tf.minimum(total_demand , avg_consumption) *\
# 				prices[tick] + \
# 				tf.maximum(0.0, demand_overflow) * \
# 				crid_price
# 	unit_price_2 = tf.where(tf.less(0.0, total_demand),
# 							tf.math.divide_no_nan(total_buy, total_demand),
# 							tf.math.divide_no_nan(total_sell, total_demand))
# 	# filter = tf.less(0.0, total_demand)
# 	# neg_filter = tf.logical_not(filter)
# 	# unit_price = tf.cast(filter, tf.float32) * tf.math.divide_no_nan(total_buy, total_demand) + \
# 	# 			 tf.cast(neg_filter, tf.float32) * tf.math.divide_no_nan(total_sell, total_demand)
# 	return unit_price_2 * action

def vanilla(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor,
            avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, _, __ = _get_basic_values(prices, crid_price, tick, action, avg_consumption)
    return (buy_price + sell_price) * action


def punishing_uniform(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor,
                      avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)
    total_ammount = total_demand * (buy_price + sell_price)
    total_prices = tf.concat((prices, [crid_price, sell_price_min]), axis=0)
    min_price = tf.math.reduce_min(total_prices)
    price_diff = (tf.math.reduce_max(total_prices) - min_price)
    punishing_coefficient_unclipped = 1.6 * tf.math.sigmoid(4 * ((buy_price_max - min_price) / price_diff - 0.5)) - 0.3
    punishing_coefficient = tf.clip_by_value(punishing_coefficient_unclipped, 0.0, 1.0)
    buyers_tensor = tf.clip_by_value(action, 0.0, 10000000.0)
    buy_ammount = tf.reduce_sum(buyers_tensor)
    sellers_tensor = tf.clip_by_value(action, -10000000.0, 0.0)
    sell_ammount = tf.reduce_sum(sellers_tensor)
    minimum_valid_buy_price = tf.math.divide_no_nan(total_ammount - sell_ammount * sell_price_min, buy_ammount)
    # final_buy_price = (buy_price_max - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_buy_price = (buy_price - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_sell_price = tf.math.divide_no_nan((total_ammount - buy_ammount * final_buy_price), sell_ammount)
    # return (tf.cast(tf.less(0.0, action), tf.float32) * final_buy_price + \
    # 	    tf.cast(tf.less(action, 0.0), tf.float32) * final_sell_price) * \
    # 	   action
    return final_buy_price * buyers_tensor + final_sell_price * sellers_tensor


def punishing_non_uniform_non_individually_rational(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor,
                                                    action: tf.Tensor, avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)
    total_ammount = total_demand * (buy_price + sell_price)
    total_prices = tf.concat((prices, [crid_price, sell_price_min]), axis=0)
    price_diff = (tf.math.reduce_max(total_prices) - tf.math.reduce_min(total_prices))
    min_price = tf.math.reduce_min(total_prices)
    punishing_coefficient_unclipped = 1.6 * tf.math.sigmoid(4 * ((buy_price_max - min_price) / price_diff - 0.5)) - 0.3
    punishing_coefficient = tf.clip_by_value(punishing_coefficient_unclipped, 0.0, 1.0)
    buyers_tensor = tf.clip_by_value(action, 0.0, 10000000.0)
    buy_ammount = tf.reduce_sum(buyers_tensor)
    sellers_tensor = tf.clip_by_value(action, -10000000.0, 0.0)
    sell_ammount = tf.reduce_sum(sellers_tensor)
    minimum_valid_buy_price = tf.math.divide_no_nan(total_ammount - sell_ammount * sell_price_min, buy_ammount)
    # final_buy_price = (buy_price_max - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_buy_price = (buy_price - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_sell_price = tf.math.divide_no_nan((total_ammount - buy_ammount * final_buy_price), sell_ammount)

    buy_ratios = tf.math.divide_no_nan(buyers_tensor, buy_ammount)
    inverse_buy_ratios = tf.math.divide_no_nan(1.0, buy_ratios)
    inverse_exp_buy_tensor = tf.exp(-inverse_buy_ratios)
    inverse_exp_buy_ratios = inverse_exp_buy_tensor / tf.reduce_sum(inverse_exp_buy_tensor)
    filtered_inverse_exp_buy_ratios = inverse_exp_buy_ratios * tf.cast(tf.less(0.0, buyers_tensor), tf.float32)
    corrected_inverse_exp_buy_ratios = tf.math.divide_no_nan(filtered_inverse_exp_buy_ratios,
                                                             tf.math.reduce_sum(filtered_inverse_exp_buy_ratios))
    exp_buy_ratios = tf.math.exp(buy_ratios) / tf.reduce_sum(tf.math.exp(buy_ratios))
    filtered_exp_buy_ratios = exp_buy_ratios * tf.cast(tf.less(0.0, buyers_tensor), tf.float32)
    corrected_exp_buy_ratios = tf.math.divide_no_nan(filtered_exp_buy_ratios,
                                                     tf.math.reduce_sum(filtered_exp_buy_ratios))
    buy_final_ratios = corrected_inverse_exp_buy_ratios * punishing_coefficient + \
                       corrected_exp_buy_ratios * (1.0 - punishing_coefficient)

    sell_ratios = tf.math.divide_no_nan(sellers_tensor, sell_ammount)
    inverse_sell_ratios = tf.math.divide_no_nan(1.0, sell_ratios)
    inverse_exp_sell_tensor = tf.exp(-inverse_sell_ratios)
    inverse_exp_sell_ratios = inverse_exp_sell_tensor / tf.reduce_sum(inverse_exp_sell_tensor)
    filtered_inverse_exp_sell_ratios = inverse_exp_sell_ratios * tf.cast(tf.less(sellers_tensor, 0.0), tf.float32)
    corrected_inverse_exp_sell_ratios = tf.math.divide_no_nan(filtered_inverse_exp_sell_ratios,
                                                              tf.math.reduce_sum(filtered_inverse_exp_sell_ratios))
    exp_sell_ratios = tf.math.exp(sell_ratios) / tf.reduce_sum(tf.math.exp(sell_ratios))
    filtered_exp_sell_ratios = exp_sell_ratios * tf.cast(tf.less(sellers_tensor, 0.0), tf.float32)
    corrected_exp_sell_ratios = tf.math.divide_no_nan(filtered_exp_sell_ratios,
                                                      tf.math.reduce_sum(filtered_exp_sell_ratios))
    sell_final_ratios = corrected_inverse_exp_sell_ratios * punishing_coefficient + \
                        corrected_exp_sell_ratios * (1.0 - punishing_coefficient)

    uniform_buy_tensor = final_buy_price * buyers_tensor
    final_buy_tensor = uniform_buy_tensor * 0.8 + buy_final_ratios * tf.reduce_sum(uniform_buy_tensor * 0.2)
    uniform_sell_tensor = final_sell_price * sellers_tensor
    final_sell_tensor = uniform_sell_tensor * 0.8 + sell_final_ratios * tf.reduce_sum(uniform_sell_tensor * 0.2)
    return final_buy_tensor + final_sell_tensor


def punishing_non_uniform_individually_rational(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor,
                                                action: tf.Tensor, avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)
    total_ammount = total_demand * (buy_price + sell_price)
    total_prices = tf.concat((prices, [crid_price, sell_price_min]), axis=0)
    price_diff = (tf.math.reduce_max(total_prices) - tf.math.reduce_min(total_prices))
    min_price = tf.math.reduce_min(total_prices)
    punishing_coefficient_unclipped = 1.6 * tf.math.sigmoid(4 * ((buy_price_max - min_price) / price_diff - 0.5)) - 0.3
    punishing_coefficient = tf.clip_by_value(punishing_coefficient_unclipped, 0.0, 1.0)
    buyers_tensor = tf.clip_by_value(action, 0.0, 10000000.0)
    buy_ammount = tf.reduce_sum(buyers_tensor)
    sellers_tensor = tf.clip_by_value(action, -10000000.0, 0.0)
    sell_ammount = tf.reduce_sum(sellers_tensor)
    minimum_valid_buy_price = tf.math.divide_no_nan(total_ammount - sell_ammount * sell_price_min, buy_ammount)
    # final_buy_price = (buy_price_max - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_buy_price = (buy_price - minimum_valid_buy_price) * punishing_coefficient + minimum_valid_buy_price
    final_sell_price = tf.math.divide_no_nan((total_ammount - buy_ammount * final_buy_price), sell_ammount)

    buy_ratios = tf.math.divide_no_nan(buyers_tensor, buy_ammount)
    inverse_buy_ratios = tf.clip_by_value(tf.math.divide_no_nan(1.0, buy_ratios), 0.0,
                                          87.0)  # clip because exp overflows
    # inverse_exp_buy_tensor = tf.exp(inverse_buy_ratios)
    inverse_exp_buy_tensor = tf.exp(inverse_buy_ratios)
    inverse_exp_buy_ratios = inverse_exp_buy_tensor / tf.reduce_sum(inverse_exp_buy_tensor)
    filtered_inverse_exp_buy_ratios = inverse_exp_buy_ratios * tf.cast(tf.less(0.0, buyers_tensor), tf.float32)
    corrected_inverse_exp_buy_ratios = tf.math.divide_no_nan(filtered_inverse_exp_buy_ratios,
                                                             tf.math.reduce_sum(filtered_inverse_exp_buy_ratios))
    exp_buy_ratios = tf.math.exp(buy_ratios) / tf.reduce_sum(tf.math.exp(buy_ratios))
    filtered_exp_buy_ratios = exp_buy_ratios * tf.cast(tf.less(0.0, buyers_tensor), tf.float32)
    corrected_exp_buy_ratios = tf.math.divide_no_nan(filtered_exp_buy_ratios,
                                                     tf.math.reduce_sum(filtered_exp_buy_ratios))
    buy_final_ratios = corrected_inverse_exp_buy_ratios * punishing_coefficient + \
                       corrected_exp_buy_ratios * (1.0 - punishing_coefficient)

    sell_ratios = tf.math.divide_no_nan(sellers_tensor, sell_ammount)
    inverse_sell_ratios = tf.math.divide_no_nan(1.0, sell_ratios)
    inverse_exp_sell_tensor = tf.exp(-inverse_sell_ratios)
    inverse_exp_sell_ratios = inverse_exp_sell_tensor / tf.reduce_sum(inverse_exp_sell_tensor)
    filtered_inverse_exp_sell_ratios = inverse_exp_sell_ratios * tf.cast(tf.less(sellers_tensor, 0.0), tf.float32)
    corrected_inverse_exp_sell_ratios = tf.math.divide_no_nan(filtered_inverse_exp_sell_ratios,
                                                              tf.math.reduce_sum(filtered_inverse_exp_sell_ratios))
    exp_sell_ratios = tf.math.exp(sell_ratios) / tf.reduce_sum(tf.math.exp(sell_ratios))
    filtered_exp_sell_ratios = exp_sell_ratios * tf.cast(tf.less(sellers_tensor, 0.0), tf.float32)
    corrected_exp_sell_ratios = tf.math.divide_no_nan(filtered_exp_sell_ratios,
                                                      tf.math.reduce_sum(filtered_exp_sell_ratios))
    sell_final_ratios = corrected_inverse_exp_sell_ratios * punishing_coefficient + \
                        corrected_exp_sell_ratios * (1.0 - punishing_coefficient)

    max_buy_ammount = buyers_tensor * buy_price_max
    final_buy_tensor = max_buy_ammount - buy_final_ratios * tf.reduce_sum(
        max_buy_ammount - final_buy_price * buyers_tensor)
    min_sell_ammount = sellers_tensor * sell_price_min
    final_sell_tensor = min_sell_ammount + sell_final_ratios * tf.reduce_sum(
        final_sell_price * sellers_tensor - min_sell_ammount)
    return final_buy_tensor + final_sell_tensor


def new_reward(prices: tf.Tensor,
               crid_price: tf.Tensor,
               tick: tf.Tensor,
               action: tf.Tensor,
               avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)
    total_ammount = total_demand * (buy_price + sell_price)

    buyers_tensor = tf.nn.relu(action)
    buy_ammount = tf.reduce_sum(buyers_tensor)

    sellers_tensor = tf.nn.relu(-action)
    sell_ammount = tf.reduce_sum(sellers_tensor)

    neutralized_load = tf.minimum(buy_ammount, sell_ammount)
    common_price = (prices[tick] + 0.8 * crid_price) / 2.0

    # buyers_indexes = tf.math.sign(buyers_tensor)
    # sellers_indexes = tf.math.sign(sellers_tensor)
    # neutralized_sell_tensor = sellers_indexes * tf.math.divide_no_nan(neutralized_load, tf.reduce_sum(sellers_indexes))

    sorted_buyers_indexes = tf.argsort(buyers_tensor)
    sorted_buyers = tf.gather(buyers_tensor, sorted_buyers_indexes)
    sorted_buyers_indexes = tf.math.sign(sorted_buyers)
    neutralized_buy_tensor = sorted_buyers_indexes * tf.math.divide_no_nan(neutralized_load,
                                                                           tf.reduce_sum(sorted_buyers_indexes))
    equally_distributed_buyers_tensor = neutralized_buy_tensor - sorted_buyers
    unclipped_sorted_buyers_tensor = tf.maximum(tf.cumsum(equally_distributed_buyers_tensor, exclusive=True),
                                                equally_distributed_buyers_tensor)
    clipped_sorted_buyers_tensor = -tf.clip_by_value(unclipped_sorted_buyers_tensor, -100000., 0.0)
    final_buyers_tensor = tf.gather(clipped_sorted_buyers_tensor, tf.argsort(sorted_buyers_indexes))
    unclipped_buyers_tensor = tf.gather(unclipped_sorted_buyers_tensor, tf.argsort(sorted_buyers_indexes))


def new_reward_proportional(prices: tf.Tensor,
                            crid_price: tf.Tensor,
                            tick: tf.Tensor,
                            action: tf.Tensor,
                            avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)

    buyers_tensor = tf.nn.relu(action)
    buy_ammount = tf.reduce_sum(buyers_tensor)

    sellers_tensor = tf.nn.relu(-action)
    sell_ammount = tf.reduce_sum(sellers_tensor)

    neutralized_load = tf.minimum(buy_ammount, sell_ammount)
    common_price = (prices[tick] + 0.8 * crid_price) / 2.0

    proportion_buyers_tensor = neutralized_load * tf.math.divide_no_nan(buyers_tensor, tf.reduce_sum(buyers_tensor))
    proportion_sellers_tensor = neutralized_load * tf.math.divide_no_nan(sellers_tensor, tf.reduce_sum(sellers_tensor))

    final_buyers_tensor = buyers_tensor - proportion_buyers_tensor
    final_sellers_tensor = sellers_tensor - proportion_sellers_tensor

    final_tensor = final_buyers_tensor * buy_price + \
                   (proportion_buyers_tensor - proportion_sellers_tensor) * common_price - \
                   final_sellers_tensor * sell_price

    return final_tensor


def new_reward_proportional_buyers_biased(prices: tf.Tensor,
                                          crid_price: tf.Tensor,
                                          tick: tf.Tensor,
                                          action: tf.Tensor,
                                          avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)

    buyers_tensor = tf.nn.relu(action)
    buy_ammount = tf.reduce_sum(buyers_tensor)

    sellers_tensor = tf.nn.relu(-action)
    sell_ammount = tf.reduce_sum(sellers_tensor)

    neutralized_load = tf.minimum(buy_ammount, sell_ammount)
    # common_price = (prices[tick] + 0.8 * crid_price) / 2.0
    common_price = 0.8 * crid_price

    proportion_buyers_tensor = neutralized_load * tf.math.divide_no_nan(buyers_tensor, tf.reduce_sum(buyers_tensor))
    proportion_sellers_tensor = neutralized_load * tf.math.divide_no_nan(sellers_tensor, tf.reduce_sum(sellers_tensor))

    final_buyers_tensor = buyers_tensor - proportion_buyers_tensor
    final_sellers_tensor = sellers_tensor - proportion_sellers_tensor

    final_tensor = final_buyers_tensor * buy_price + \
                   (proportion_buyers_tensor - proportion_sellers_tensor) * common_price - \
                   final_sellers_tensor * sell_price

    return final_tensor


def new_reward_proportional_sellers_biased(prices: tf.Tensor,
                                          crid_price: tf.Tensor,
                                          tick: tf.Tensor,
                                          action: tf.Tensor,
                                          avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)

    buyers_tensor = tf.nn.relu(action)
    buy_ammount = tf.reduce_sum(buyers_tensor)

    sellers_tensor = tf.nn.relu(-action)
    sell_ammount = tf.reduce_sum(sellers_tensor)

    neutralized_load = tf.minimum(buy_ammount, sell_ammount)
    # common_price = (prices[tick] + 0.8 * crid_price) / 2.0
    common_price = prices[tick]

    proportion_buyers_tensor = neutralized_load * tf.math.divide_no_nan(buyers_tensor, tf.reduce_sum(buyers_tensor))
    proportion_sellers_tensor = neutralized_load * tf.math.divide_no_nan(sellers_tensor, tf.reduce_sum(sellers_tensor))

    final_buyers_tensor = buyers_tensor - proportion_buyers_tensor
    final_sellers_tensor = sellers_tensor - proportion_sellers_tensor

    final_tensor = final_buyers_tensor * buy_price + \
                   (proportion_buyers_tensor - proportion_sellers_tensor) * common_price - \
                   final_sellers_tensor * sell_price

    return final_tensor



def new_reward_vcg_like(prices: tf.Tensor,
                        crid_price: tf.Tensor,
                        tick: tf.Tensor,
                        action: tf.Tensor,
                        avg_consumption: tf.Tensor) -> tf.Tensor:
    vanilla_tensor = vanilla(prices, crid_price, tick, action, avg_consumption)
    return tf.ones_like(action) * tf.reduce_mean(vanilla_tensor)


def extreme(prices: tf.Tensor,
            crid_price: tf.Tensor,
            tick: tf.Tensor,
            action: tf.Tensor,
            avg_consumption: tf.Tensor) -> tf.Tensor:
    total_demand, buy_price, sell_price, sell_price_min, buy_price_max = _get_basic_values(prices,
                                                                                           crid_price,
                                                                                           tick,
                                                                                           action,
                                                                                           avg_consumption)
