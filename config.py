# Parking Space Variables
# Charging rates in kW
from functools import reduce

DISCOUNT = 1.0
VEHICLE_BATTERY_CAPACITY = 60.0
VEHICLE_MIN_CHARGE = 0.0
MAX_CHARGING_RATE = 20.0
MAX_DISCHARGING_RATE = 20.0
AVG_CHARGING_RATE = 10
BATTERY_CAPACITY = 60

NUM_OF_ACTIONS = 24

NUMBER_OF_AGENTS = 12
MAX_BUFFER_SIZE = 100000

CRID_COEFFICIENT = 0.8
GARAGE_LIST = []
# AVG_CONSUMPTION = lambda : reduce(lambda x, y: x + y, GARAGE_LIST, 0.0) * AVG_CHARGING_RATE

# TRAIN_ITERATIONS = 24 * 100 * 500
TRAIN_ITERATIONS = 500

# USE_JIT = True
USE_JIT = False


# EAGER_EXECUTION = True
EAGER_EXECUTION = False

KEEP_BEST = True
# KEEP_BEST = False

START_FROM_BEST = True
# START_FROM_BEST = False

