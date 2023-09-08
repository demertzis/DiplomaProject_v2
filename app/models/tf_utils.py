import tensorflow as tf


# @tf.function()
def my_round(x: tf.Tensor, decimals: tf.Tensor):
    # print("Tracing my_round")
    mult = tf.math.pow(tf.constant(10.0), tf.cast(decimals, tf.float32))
    return tf.round(tf.cast(x, dtype=tf.float32) * mult) / mult


# @tf.function(input_signature=[tf.TensorSpec([None], tf.float32),
#                               tf.TensorSpec([], tf.int32)])
def my_round_vectorized(x: tf.Tensor, decimals: tf.Tensor):
    # print("Tracing my_round")
    mult = tf.math.pow(tf.constant(10.0), tf.cast(decimals, tf.float32))
    return tf.round(tf.cast(x, dtype=tf.float32) * mult) / mult


def my_round_16(x: tf.Tensor, decimals: tf.Tensor):
    # print("Tracing my_round")
    mult = tf.math.pow(tf.constant(10.0, tf.float16), tf.cast(decimals, tf.float16))
    return tf.round(tf.cast(x, dtype=tf.float16) * mult) / mult


def my_pad(t: tf.Tensor) -> tf.Tensor:
    s = tf.shape(t)
    paddings = [[0, 13 - s[..., -1]]]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=t[..., -1])


def tf_find_intersection_v2(x1, y1, c) -> tf.Tensor:
    diff = y1[..., :-1] - tf.fill(tf.shape(y1[..., :-1]), c)
    diff_plus_one = y1[..., 1:] - tf.fill(tf.shape(y1[..., 1:]), c)
    transition_tensor = (diff_plus_one * diff)
    intersection_tensor = tf.where(tf.less_equal(transition_tensor, 0.0))

    final_tensor = tf.gather(x1, intersection_tensor) + \
                   tf.math.divide_no_nan(tf.gather(tf.math.abs(diff), intersection_tensor),
                                         (tf.gather(tf.math.abs(diff), intersection_tensor) +
                                          tf.gather(tf.math.abs(diff_plus_one), intersection_tensor)))

    return tf.squeeze(final_tensor[:2][-1])

    # maybe_extra_key = tf.cond(tf.equal((diff - diff_plus_one)[..., -1], 0.0),
    #                           lambda: tf.squeeze(x1[..., -1]),
    #                           lambda: tf.constant())

    # return tf.gather(x1, intersection_tensor + 1)


@tf.function(input_signature=[tf.TensorSpec([13, ], tf.float32),
                              tf.TensorSpec([None, 13], tf.float32),
                              tf.TensorSpec([None, 1], tf.float32)])
def tf_find_intersection_vectorized(x1, y1, c) -> tf.Tensor:
    # print('Tracing vectorized_tf_find_intersection_vectorized')
    diff = y1[..., :-1] - c
    diff_plus_one = y1[..., 1:] - c
    transition_tensor = (diff_plus_one * diff)
    intersection_tensor = tf.where(tf.less_equal(transition_tensor, 0.0))
    sparse_indexes = tf.sparse.SparseTensor(tf.cast(intersection_tensor, tf.int64),
                                            tf.ones(tf.shape(intersection_tensor)[:-1], tf.int32),
                                            tf.cast(tf.shape(transition_tensor), tf.int64))
    final_indexes = tf.cast(tf.math.argmax(tf.sparse.to_dense(sparse_indexes)[..., 1:], axis=1, output_type=tf.int32),
                            tf.float32)
    gather_tensor = tf.cast(final_indexes + tf.math.divide_no_nan(final_indexes, final_indexes), tf.int32)
    gather_nd_tensor = tf.expand_dims(gather_tensor, axis=1)
    gather_nd_tensor = tf.concat(
        (tf.expand_dims(tf.range(tf.shape(diff)[0], dtype=tf.int32), axis=1), gather_nd_tensor), axis=1)
    # gather_nd_tensor b=
    final_tensor = tf.gather(x1, gather_tensor) + \
                   tf.math.divide_no_nan(tf.gather_nd(tf.math.abs(diff), gather_nd_tensor),
                                         (tf.gather_nd(tf.math.abs(diff), gather_nd_tensor) +
                                          tf.gather_nd(tf.math.abs(diff_plus_one), gather_nd_tensor)))
    return final_tensor

    # diff = y1[..., :-1] - c
    # diff_plus_one = y1[..., 1:] - c
    # transition_tensor = (diff_plus_one * diff)[..., 1:]
    # intersection_tensor = tf.where(tf.less_equal(transition_tensor , 0.0),
    #                                1.0,
    #                                0.0)
    # final_indexes = tf.math.argmax(intersection_tensor, axis=1, output_type=tf.int32)
    # gather_tensor = tf.where(tf.less(0, final_indexes), final_indexes + 1, 0)
    # indexes = tf.stack((tf.transpose(tf.stack((tf.zeros(gather_tensor.shape[1], tf.int32),
    #                                            tf.range(gather_tensor.shape[1])), 0)),
    #                     tf.transpose(tf.stack((tf.ones(gather_tensor.shape[1], tf.int32),
    #                                            tf.range(gather_tensor.shape[1])), 0))),
    #                    0)
    # gather_nd_tensor = tf.concat((indexes, tf.expand_dims(gather_tensor, 2)), 2)
    # diff_gathered = tf.gather_nd(tf.math.abs(diff), gather_nd_tensor)
    # diff_plus_one_gathered = tf.gather_nd(tf.math.abs(diff_plus_one), gather_nd_tensor)
    # final_tensor = tf.gather(x1, gather_tensor) +\
    #                tf.math.divide_no_nan(diff_gathered, diff_gathered + diff_plus_one_gathered)
    # return final_tensor


zero_16 = tf.constant(0.0, tf.float16)
one_16 = tf.constant(1.0, tf.float16)


def tf_find_intersection_vectorized_f16(x1, y1, c, num_of_vehicles) -> tf.Tensor:
    # print('Tracing vectorized_tf_find_intersection_vectorized')
    diff = y1[..., :-1] - c
    diff_plus_one = y1[..., 1:] - c
    transition_tensor = (diff_plus_one * diff)[..., 1:]
    # transition_tensor = (diff_plus_one * diff) + tf.constant([[1.0] + [0.0] * 11], tf.float16)
    intersection_tensor = tf.where(tf.less_equal(transition_tensor, zero_16),
                                   one_16,
                                   zero_16)
    # final_indexes = tf.math.argmax(intersection_tensor, axis=2, output_type=tf.int32)
    # gather_tensor = tf.where(tf.less(0, final_indexes), final_indexes + 1, 0)
    final_indexes = tf.math.argmax(intersection_tensor, axis=2, output_type=tf.int32) + 1
    gather_tensor = tf.where(tf.logical_and(tf.equal(final_indexes, 1),
                                            tf.logical_or(tf.equal(diff[..., 1], zero_16),
                                                          tf.less(zero_16, transition_tensor[..., 0])), ),
                             0,
                             final_indexes)
    indexes = tf.stack((tf.transpose(tf.stack((tf.zeros(num_of_vehicles, tf.int32),
                                               tf.range(num_of_vehicles)), 0)),
                        tf.transpose(tf.stack((tf.ones(num_of_vehicles, tf.int32),
                                               tf.range(num_of_vehicles)), 0))),
                       0)
    gather_nd_tensor = tf.concat((indexes, tf.expand_dims(gather_tensor, 2)), 2)
    diff_gathered = tf.gather_nd(tf.math.abs(diff), gather_nd_tensor)
    diff_plus_one_gathered = tf.gather_nd(tf.math.abs(diff_plus_one), gather_nd_tensor)
    final_tensor = tf.gather(x1, gather_tensor) + \
                   tf.math.divide_no_nan(diff_gathered, diff_gathered + diff_plus_one_gathered)
    return final_tensor


def tf_find_intersection_vectorized_single_model(x1, y1, c, num_of_vehicles, num_of_agents) -> tf.Tensor:
    # print('Tracing vectorized_tf_find_intersection_vectorized')
    diff = y1[..., :-1] - c
    diff_plus_one = y1[..., 1:] - c
    transition_tensor = (diff_plus_one * diff)[..., 1:]
    # intersection_tensor = tf.where(tf.less_equal(transition_tensor , 0.0),
    #                                1.0,
    #                                0.0)
    intersection_tensor = tf.cast(tf.less_equal(transition_tensor, 0.0), tf.float32)
    final_indexes = tf.math.argmax(intersection_tensor, axis=-1, output_type=tf.int32) + 1
    # gather_tensor_2 = tf.where(tf.logical_and(tf.equal(final_indexes, 1),
    #                                           tf.logical_or(tf.equal(diff[..., 1], 0.0),
    #                                                          tf.less(0.0, transition_tensor[..., 0])),),
    #                            0,
    #                            final_indexes)
    mask = tf.logical_or(tf.not_equal(final_indexes, 1),
                         tf.logical_and(tf.not_equal(diff[..., 1], 0.0),
                                        tf.less_equal(transition_tensor[..., 0], 0.0)))
    gather_tensor = tf.cast(mask, tf.int32) * final_indexes
    coordinates = tf.meshgrid(tf.range(2), tf.range(num_of_agents), tf.range(num_of_vehicles), indexing='ij')
    new_gather_tensor = tf.stack(coordinates + [gather_tensor], axis=-1)
    diff_gathered = tf.gather_nd(tf.math.abs(diff), new_gather_tensor)
    diff_plus_one_gathered = tf.gather_nd(tf.math.abs(diff_plus_one), new_gather_tensor)
    final_tensor = tf.gather(x1, gather_tensor) + \
                   tf.math.divide_no_nan(diff_gathered, diff_gathered + diff_plus_one_gathered)
    return final_tensor


def tf_find_intersection_vectorized_f32(x1, y1, c, num_of_vehicles) -> tf.Tensor:
    # print('Tracing vectorized_tf_find_intersection_vectorized')
    diff = y1[..., :-1] - c
    diff_plus_one = y1[..., 1:] - c
    transition_tensor = (diff_plus_one * diff)[..., 1:]
    # intersection_tensor = tf.where(tf.less_equal(transition_tensor , 0.0),
    #                                1.0,
    #                                0.0)
    intersection_tensor = tf.cast(tf.less_equal(transition_tensor, 0.0), tf.float32)
    final_indexes = tf.math.argmax(intersection_tensor, axis=2, output_type=tf.int32) + 1
    # gather_tensor_2 = tf.where(tf.logical_and(tf.equal(final_indexes, 1),
    #                                           tf.logical_or(tf.equal(diff[..., 1], 0.0),
    #                                                          tf.less(0.0, transition_tensor[..., 0])),),
    #                            0,
    #                            final_indexes)
    mask = tf.logical_or(tf.not_equal(final_indexes, 1),
                         tf.logical_and(tf.not_equal(diff[..., 1], 0.0),
                                        tf.less_equal(transition_tensor[..., 0], 0.0)))
    gather_tensor = tf.cast(mask, tf.int32) * final_indexes
    indexes = tf.stack((tf.transpose(tf.stack((tf.zeros(num_of_vehicles, tf.int32),
                                               tf.range(num_of_vehicles)), 0)),
                        tf.transpose(tf.stack((tf.ones(num_of_vehicles, tf.int32),
                                               tf.range(num_of_vehicles)), 0))),
                       0)
    # tf.meshgrid
    gather_nd_tensor = tf.concat((indexes, tf.expand_dims(gather_tensor, 2)), 2)
    diff_gathered = tf.gather_nd(tf.math.abs(diff), gather_nd_tensor)
    diff_plus_one_gathered = tf.gather_nd(tf.math.abs(diff_plus_one), gather_nd_tensor)
    final_tensor = tf.gather(x1, gather_tensor) + \
                   tf.math.divide_no_nan(diff_gathered, diff_gathered + diff_plus_one_gathered)
    return final_tensor
