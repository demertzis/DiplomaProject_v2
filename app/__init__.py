# from gym.envs.registration import register
# from config import ENERGY_CURVE_TRAIN, ENERGY_CURVE_EVAL
#
# # from . import utilities
# # from app.models.power_energy_curve import powerMarketEnv
#
# from .models.energy import EnergyCurve
#
# # DAY_AHEAD_EXAMPLE =
#
# # df = deepcopy(utilities.get_data())
#
#
# energy_curve_train = ENERGY_CURVE_TRAIN
# energy_curve_eval = ENERGY_CURVE_EVAL
#
# register(
#     id='PowerTrain-v0',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_train,
#         'reward_function': 0
#     }
# )
#
# register(
#     id='PowerEval-v0',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_eval,
#         'reward_function': 0
#     }
# )
#
# register(
#     id='PowerTrain-v1',
#     entry_point='app.models.power_market_env:powerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_train,
#         'reward_function': 1
#     }
# )
# register(
#     id='PowerEval-v1',
#     entry_point='app.models.Power_market_env:powerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_eval,
#         'reward_function': 1
#     }
# )
#
# register(
#     id='PowerTrain-v2',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_train,
#         'reward_function': 2
#     }
# )
# register(
#     id='PowerEval-v2',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_eval,
#         'reward_function': 2
#     }
# )
#
# register(
#     id='PowerTrain-v3',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_train,
#         'reward_function': 3
#     }
# )
# register(
#     id='PowerEval-v3',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_eval,
#         'reward_function': 3
#     }
# )
#
# register(
#     id='PowerTrain-v4',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_train,
#         'reward_function': 4
#     }
# )
# register(
#     id='PowerEval-v4',
#     entry_point='app.models.power_market_env:PowerMarketEnv',
#     kwargs={
#         'energy_curve': energy_curve_eval,
#         'reward_function': 4
#     }
# )
