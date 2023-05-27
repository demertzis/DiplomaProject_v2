import tensorflow as tf
from config import CRID_COEFFICIENT, AVG_CONSUMPTION

def vanilla(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor, avg_consumption: tf.Tensor) -> tf.Tensor:
	total_demand = tf.math.reduce_sum(action)
	demand_overflow = total_demand - avg_consumption
	sell_price_min = crid_price * CRID_COEFFICIENT
	total_sell = total_demand * sell_price_min
	total_buy = tf.minimum(total_demand , avg_consumption) *\
				prices[tick] + \
				tf.maximum(0.0, demand_overflow) * \
				crid_price
	unit_price_2 = tf.where(tf.less(0.0, total_demand),
							tf.math.divide_no_nan(total_buy, total_demand),
							tf.math.divide_no_nan(total_sell, total_demand))
	return unit_price_2 * action