import math
import random
from typing import List, Tuple, Union
from intersect import intersection
import numpy as np

from config import AVG_CHARGING_RATE

def find_intersection(x1, y1, x2, y2) -> Union[Tuple[float, float], None]:
    x, y = intersection(x1, y1, x2, y2)
    intersections = list(filter(lambda val: val[0] > 0, set(zip(x, y))))

    if len(intersections) == 0:
        return None

    return intersections[0]


def find_intersection_v2(x1, y1, c) -> Union[Tuple[float, float], None]:
    length = len(y1)
    for i in range(length - 2, -1, -1):
        if y1[i + 1] == c:
            return x1[i + 1], c
        elif y1[i] > c > y1[i + 1] or y1[i] < c < y1[i + 1]:
            y = abs(y1[i] - y1[i + 1])
            y_ = abs(c - y1[i + 1])
            x = abs(x1[i + 1] - x1[i])
            x_ = (1 - y_ / y) * x
            return x1[i] + x_, c

    return None


class VehicleDistributionList(list):
    def __init__(self, avg_list: List[float] = [0.0] * 24, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_consumption_list = avg_list


def create_vehicle_distribution(steps: object = 24 * 30 * 6) -> List[List[int]]:
    if steps % 24 != 0:
        raise
    time_of_day = 0
    vehicles = VehicleDistributionList([0.0] * 24, [])
    for _ in range(steps):
        day_coefficient = math.sin(math.pi / 6 * time_of_day) / 2 + 0.5
        new_cars = max(0, int(np.random.normal(20 * day_coefficient, 2 * day_coefficient)))
        time_of_day += 1
        time_of_day %= 24
        if time_of_day > 21:
            vehicles.append([])
        else:
            vehicles.append(
                [
                    (
                        min(
                            25 - time_of_day,  # think it's better, allows to have vehicles at 23:00
                            random.randint(7, 12),
                        ),
                        round(6 + random.random() * 20, 2),
                        round(34 + random.random() * 20, 2),
                    )
                    for _ in range(new_cars)
                ]
            )
#TODO fix compute of avg_consumption_list
    total_days = len(vehicles) // 24
    for i in range(len(vehicles) - len(vehicles) % 24):
        # vehicles.avg_consumption_list[i % 24] += len(vehicles[i]) * AVG_CHARGING_RATE
        for v in vehicles[i]:
            for z in range(v[0]):
                vehicles.avg_consumption_list[(i + 1) % 24 + z - 1] += 1

    vehicles.avg_consumption_list = [x / total_days * AVG_CHARGING_RATE for x in vehicles.avg_consumption_list]

    return vehicles
