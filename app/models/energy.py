import datetime
import random
from typing import List, Tuple
import numpy as np


class EnergyCurve:
    """
    A class that holds the energy curve along with its derivatives

    ### Arguments:
        data (`Tuple(datetime, float)[]`) :
            description: The raw data array. Each point consists of the timestamp and the cost of energy
    * Changed the courve in order to contain the intra day price
    """
    _y: List[float]
    _start: int = 0
    _end: int = 24

    def __init__(self, dataFile: str, name: str):
        self._data: List[Tuple[datetime.datetime, float, float]] = []
        self._x: List[datetime.datetime] = []
        self._raw_y: List[List[Tuple[float, float]]] = [[]]
        self._y: List[Tuple[float, float]] = []
        self._normalizing_coefficient = 0.0
        with open(dataFile) as csv:
            counter = 0
            for line in csv.readlines():
                date, value, value_intra_day = line.split(",")
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                value = float(value)
                value_intra_day = float(value_intra_day)
                self._normalizing_coefficient = max(value, value_intra_day, self._normalizing_coefficient)
                self._data.append((date, value, value_intra_day))
                self._x.append(date)
                if counter == 24:
                    counter = 0
                    self._raw_y.append([])
                self._raw_y[-1].append((value, value_intra_day))
                counter += 1
        self.name = name
        self.randomize_data(self.name == 'eval')

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

    def get_current_batch(self, normalized=False):
        """
        Returns the current batch of values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        return [val / (self._normalizing_coefficient if normalized else 1) for val, _ in self._y[self._start:self._end]]

    def get_current_batch_intra_day(self, normalized=False):
        """
        Returns the current batch of values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        return [val / (self._normalizing_coefficient if normalized else 1) for _, val in self._y[self._start:self._end]]


    def get_next_batch(self, normalized=False):
        """
        Returns the next batch of values
        Moves window to the next values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        ret = [val / (self._normalizing_coefficient if normalized else 1) for val, _ in self._y[self._start:self._end]]
        if len(self._data) > self._end:
            self._start += 1
            self._end += 1
        else:
            self.reset()

        return ret

    def get_next_batch_intra_day(self, normalized=False):
        """
        Returns the next batch of values
        Moves window to the next values

        ### Arguments
            bool : Whether to normalize the data or not

        ### Returns
            float[24] : A 24 size array with the energy cost
        """
        ret = [val / (self._normalizing_coefficient if normalized else 1) for _, val in self._y[self._start:self._end]]
        if len(self._data) > self._end:
            self._start += 1
            self._end += 1
        else:
            self.reset()

        return ret

    def get_next_episode(self):

        if len(self._data) > self._end + 24:
            self._start += 24
            self._end += 24
        else:
            self.reset()

    def reset(self):
        self._start = 0
        self._end = 24
        self.randomize_data(self.name == 'eval')

    def get_current_cost(self):
        """
        Get current energy cost
        """
        return self._y[self._start - 1][0]

    def get_current_cost_intra_day(self):
        """
        Get current energy cost
        """
        return self._y[self._start - 1][1]

    def randomize_data(self, is_eval):
        if is_eval:
            self._y = [(val, val_intra_day) for _, val, val_intra_day in self._data]
        else:
            random.shuffle(self._raw_y)
            self._y = [item for sublist in self._raw_y for item in sublist]
