import random

from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation
from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
# from FleetRL.utils.prices import load_prices

env = FleetEnv()
env.reset()
for i in range(2):
    action = []
    # print(env.step([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    for j in range(env.num_cars):
        action.append(random.random())
    print(env.step(action))

print(env.episode.reward_history)
