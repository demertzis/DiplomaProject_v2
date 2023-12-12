import datetime

import numpy as np
import tensorflow as tf

from app.policies.tf_reward_functions import new_reward_proportional

tf.config.run_functions_eagerly(True)

data = []
raw_y = [[]]
x = []
with open('data/randomized_data.csv') as csv:
    counter = 0
    max_val = 0.0
    min_val = 1000.0
    for line in csv.readlines():
        date, value, value_intra_day = line.split(",")
        date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        value = float(value)
        value_intra_day = float(value_intra_day)
        # self._normalizing_coefficient = max(value, value_intra_day, self._normalizing_coefficient)
        max_val = max(value, value_intra_day, max_val)
        min_val = min(value, value_intra_day, min_val)
        data.append((date, value, value_intra_day))
        x.append(date)
        if counter == 24:
            counter = 0
            raw_y.append([])
        raw_y[-1].append((value, value_intra_day))
        counter += 1


print('')