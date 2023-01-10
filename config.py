# Parking Space Variables
# Charging rates in kW
from functools import reduce

MAX_CHARGING_RATE = 60
MAX_DISCHARGING_RATE = 60
AVG_CHARGING_RATE = 5
BATTERY_CAPACITY = 60
NUMBER_OF_AGENTS = 1
CRID_COEFFICIENT = 0.8
GARAGE_LIST = []
AVG_CONSUMPTION = lambda : reduce(lambda x, y: x + y, GARAGE_LIST, 0.0)


