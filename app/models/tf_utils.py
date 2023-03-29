import tensorflow as tf

@tf.function()
def my_round(x: tf.Tensor, decimals: tf.Tensor):
    # print("Tracing my_round")
    mult = tf.math.pow(tf.constant(10.0), tf.cast(decimals, tf.float32))
    return tf.round(tf.cast(x, dtype=tf.float32) * mult) / mult

@tf.function(input_signature=[tf.TensorSpec([None], tf.float32),
                              tf.TensorSpec([], tf.int32)])
def my_round_vectorized(x: tf.Tensor, decimals: tf.Tensor):
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
@tf.function(input_signature=[tf.TensorSpec([13,], tf.float32),
                              tf.TensorSpec([None, 13], tf.float32),
                              tf.TensorSpec([None,1], tf.float32)])
def tf_find_intersection_vectorized(x1, y1, c) -> tf.Tensor:
    print('Tracing vectorized_tf_find_intersection_vectorized')
    diff = y1[..., :-1] - c
    diff_plus_one = y1[..., 1:] - c
    transition_tensor = (diff_plus_one * diff)
    intersection_tensor = tf.where(tf.less_equal(transition_tensor ,0.0))
    sparse_indexes = tf.sparse.SparseTensor(tf.cast(intersection_tensor, tf.int64),
                                            tf.ones(tf.shape(intersection_tensor)[:-1], tf.int32),
                                            tf.cast(tf.shape(transition_tensor), tf.int64))
    final_indexes = tf.cast(tf.math.argmax(tf.sparse.to_dense(sparse_indexes)[..., 1:], axis=1, output_type=tf.int32),
                            tf.float32)
    gather_tensor = tf.cast(final_indexes + tf.math.divide_no_nan(final_indexes, final_indexes), tf.int32)
    gather_nd_tensor = tf.expand_dims(gather_tensor, axis=1)
    gather_nd_tensor = tf.concat((tf.expand_dims(tf.range(tf.shape(diff)[0], dtype=tf.int32), axis=1), gather_nd_tensor), axis=1)
    # gather_nd_tensor b=
    final_tensor = tf.gather(x1, gather_tensor) +\
                   tf.math.divide_no_nan(tf.gather_nd(tf.math.abs(diff), gather_nd_tensor),
                                         (tf.gather_nd(tf.math.abs(diff), gather_nd_tensor) +
                                          tf.gather_nd(tf.math.abs(diff_plus_one), gather_nd_tensor)))

    return final_tensor

    # maybe_extra_key = tf.cond(tf.equal((diff - diff_plus_one)[..., -1], 0.0),
    #                           lambda: tf.squeeze(x1[..., -1]),
    #                           lambda: tf.constant())

    # return tf.gather(x1, intersection_tensor + 1)