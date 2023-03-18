from typing import Union, Tuple

import tensorflow as tf

def tf_find_intersection_v2(x1, y1, c) -> Union[Tuple[tf.Tensor, tf.Tensor], None]:
    x1 = tf.squeeze(x1)
    y1 = tf.squeeze(y1)
    c = tf.squeeze(c)
    length = y1.shape
    for i in tf.range(length - 2, -1, -1):
        if y1[i + 1] == c:
            return x1[i + 1], c
        elif tf.less(y1[i+1], tf.less(c, y1[i])) or tf.less(y1[i], tf.less(c, y1[i+1])):
            y = tf.abs(y1[i] - y1[i + 1])
            y_ = tf.abs(c - y1[i + 1])
            x = tf.abs(x1[i + 1] - x1[i])
            x_ = (1 - y_ / y) * x
            return x1[i] + x_, c

    return None
