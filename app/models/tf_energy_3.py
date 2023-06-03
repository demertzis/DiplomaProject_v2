import datetime
import random
from typing import List, Tuple
import tensorflow as tf


class EnergyCurve(tf.Module):
    """
    A class that holds the energy curve along with its derivatives

    ### Arguments:
        data (`Tuple(datetime, float)[]`) :
            description: The raw data array. Each point consists of the timestamp and the cost of energy
    * Changed the curve in order to contain the intra - day price
    """

    def __init__(self, dataFile: str, name: str):
        self._data: List[Tuple[datetime.datetime, float, float]] = []
        self._x: List[datetime.datetime] = []
        self._raw_y: List[List[Tuple[float, float]]] = [[]]
        # self._y: List[Tuple[float, float]] = []
        self._normalizing_coefficient = 0.0
        max_val = 0.0
        min_val = 1000.0
        with open(dataFile) as csv:
            counter = 0
            for line in csv.readlines():
                date, value, value_intra_day = line.split(",")
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                value = float(value)
                value_intra_day = float(value_intra_day)
                # self._normalizing_coefficient = max(value, value_intra_day, self._normalizing_coefficient)
                max_val = max(value, value_intra_day, max_val)
                min_val = min(value, value_intra_day, min_val)
                self._data.append((date, value, value_intra_day))
                self._x.append(date)
                if counter == 24:
                    counter = 0
                    self._raw_y.append([])
                self._raw_y[-1].append((value, value_intra_day))
                counter += 1
        if len(self._data) % 24 != 0:
            raise Exception('Energy price data contain incomplete day')
        self._name = name
        self._raw_y = tf.convert_to_tensor(self._raw_y)
        self._y = tf.Variable([[x,y] for z, x, y in self._data], trainable=False)
        # self._normalizing_coefficient = tf.convert_to_tensor(self._normalizing_coefficient)
        self.randomize_data(self._name == 'eval')
        self._start = tf.Variable(-24, dtype=tf.int64, trainable=False)
        self._normalizing_coefficient = max_val - min_val
        self._min_val = min_val

    def total_episodes(self):
        return len(self._data) // 24

    def get_raw_data(self):
        """
        Get the raw energy data

        ### Returns
            Tuple(datetime, float)[] : The raw energy data
        """
        return self._data

    def get_x(self):
        """
        Get the x values (datetimes)

        ### Returns
            datetime[] : The x values of the data
        """
        return self._x

    def get_y(self):
        """
        Get the energy cost values throughout time

        ### Returns:
            float[] : The y values of the data
        """
        return self._y

    def get_current_batch(self, normalized: bool = False):
        """
        Returns the current batch of values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        #print('Tracing get_current_batch')
        start = self._start
        return_tensor = self._y[start:start + 24, 0]
        # tf.print(tf.shape(return_tensor))
        if normalized:
            # return return_tensor / self._normalizing_coefficient
            return (return_tensor - self._min_val) / self._normalizing_coefficient
        else:
            return return_tensor

    def get_current_batch_intra_day(self, normalized: bool = False):
        """
        Returns the current batch of values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        #print('Tracing get_current_batch_intra_day')
        start = self._start
        return_tensor = self._y[start:start + 24, 1]
        # tf.print(tf.shape(return_tensor))
        if normalized:
            return (return_tensor - self._min_val) / self._normalizing_coefficient
        else:
            return return_tensor

    def get_next_batch(self, normalized: bool = False):
        """
        Returns the next batch of values
        Moves window to the next values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        start = self._start
        return_tensor = self._y.gather_nd(tf.reshape(tf.range(start, start + 24), [-1, 1]))[..., 0]
        if len(self._data) > start + 24:
            self._start.assign_add(1)
        else:
            self.reset()
        if normalized:
            return (return_tensor - self._min_val) / self._normalizing_coefficient
        else:
            return return_tensor



    def get_next_batch_intra_day(self, normalized: bool = False):
        """
        Returns the next batch of values
        Moves window to the next values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        start = self._start
        return_tensor = self._y.gather_nd(tf.reshape(tf.range(start, start + 24), [-1, 1]))[..., 1]
        if len(self._data) > start + 24:
            self._start.assign_add(1)
        else:
            self.reset()
        if normalized:
            return (return_tensor - self._min_val) / self._normalizing_coefficient
        else:
            return return_tensor

    # @tf.function
    def get_next_episode(self):
        #print('Tracing get_next_episode')
        new_start = (self._start + 24) % len(self._data)
        self._start.assign(new_start)

    @tf.function
    def reset(self):
        #print('Tracing reset')
        self.randomize_data(self._name == 'eval')
        self._start.assign(-24)

    def get_current_cost(self):
        """
        Get current energy cost
        """
        return self._y[self._start][0]

    def get_current_cost_intra_day(self):
        """
        Get current energy cost
        """
        return self._y[self._start][1]

    def randomize_data(self, is_eval):
        #print('Tracing randomize_data')
        # if is_eval:
        #     # self._y.assign([(val, val_intra_day) for _, val, val_intra_day in self._data])
        #     data = [[x,y] for z, x, y in self._data]
        #     self._y.assign(data)
        # else:
        if not is_eval:
            shuffled_data = tf.random.shuffle(self._raw_y)
            # self._y = [item for sublist in self._raw_y for item in sublist]
            self._y.assign(tf.reshape(shuffled_data, [-1, 2]))
