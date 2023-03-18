import tensorflow as tf

@tf.function
def my_round(x: tf.Tensor, decimals: tf.Tensor):
    # print("Tracing my_round")
    mult = tf.math.pow(tf.constant(10.0), tf.cast(decimals, tf.float32))
    return tf.round(tf.cast(x, dtype=tf.float32) * mult) / mult

def my_pad(t: tf.Tensor) -> tf.Tensor:
    s = tf.shape(t)
    paddings = [[0, 13 - s[..., -1]]]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=t[..., -1])

def tf_find_intersection_v2(x1, y1, c) -> tf.Tensor:
    diff = y1[..., :-1] - tf.fill(tf.shape(y1[..., :-1]), c)
    diff_plus_one = y1[..., 1:] - tf.fill(tf.shape(y1[..., 1:]), c)
    transition_tensor = (diff_plus_one * diff)
    intersection_tensor = tf.where(tf.less_equal(transition_tensor ,0.0))

    final_tensor = tf.gather(x1, intersection_tensor) +\
                   tf.math.divide_no_nan(tf.gather(tf.math.abs(diff), intersection_tensor),
                                         (tf.gather(tf.math.abs(diff), intersection_tensor) +
                                          tf.gather(tf.math.abs(diff_plus_one), intersection_tensor)))

    return tf.squeeze(final_tensor[:2][-1])

    # maybe_extra_key = tf.cond(tf.equal((diff - diff_plus_one)[..., -1], 0.0),
    #                           lambda: tf.squeeze(x1[..., -1]),
    #                           lambda: tf.constant())

    # return tf.gather(x1, intersection_tensor + 1)