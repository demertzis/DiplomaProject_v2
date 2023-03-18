import tensorflow as tf
from config import CRID_COEFFICIENT, AVG_CONSUMPTION


def vanilla(prices: tf.Tensor, crid_price: tf.Tensor, tick: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
	avg_consumption = tf.constant(AVG_CONSUMPTION(), dtype=tf.float32)
	total_demand = tf.squeeze(tf.math.reduce_sum(action))
	demand_overflow = total_demand - avg_consumption
	sell_price_min = crid_price * CRID_COEFFICIENT
	demand_total_price = tf.cond(tf.math.less_equal(total_demand, 0.0),
								 lambda: total_demand * sell_price_min,
								 lambda: (tf.math.minimum(total_demand , avg_consumption) *\
									 prices[tick]) + \
									 (tf.math.maximum(0.0, demand_overflow) * \
										 crid_price)
								)
	unit_price = (demand_total_price / total_demand) if tf.math.abs(total_demand) > 0.0 else tf.constant(0.0)
	return unit_price * action

	# return unit_price * tf.cond(tf.equal(tf.rank(tf.squeeze(action)), 0),
	# 							lambda: tf.expand_dims(tf.squeeze(action), 0),
	# 							lambda: tf.squeeze(action))