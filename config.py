# Parking Space Variables
# Charging rates in kW
from functools import reduce

DISCOUNT = 0.90
VEHICLE_BATTERY_CAPACITY = 60.0
MAX_CHARGING_RATE = 60
MAX_DISCHARGING_RATE = 60
AVG_CHARGING_RATE = 10
BATTERY_CAPACITY = 60
NUMBER_OF_AGENTS = 1
CRID_COEFFICIENT = 0.8
GARAGE_LIST = []
MAX_BUFFER_SIZE = 100000
AVG_CONSUMPTION = lambda : reduce(lambda x, y: x + y, GARAGE_LIST, 0.0) * AVG_CHARGING_RATE

TRAIN_ITERATIONS = 24 * 100 * 2000


