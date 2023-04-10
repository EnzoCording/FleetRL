import random

from FleetRL.env.fleet_environment import FleetEnv
from FleetRL.utils.battery_depreciation.battery_dep_base import BatteryDepreciationBase
from FleetRL.utils.battery_depreciation.my_battery_depreciation import MyBatteryDepreciation
from FleetRL.utils.prices import load_prices

env = FleetEnv()
env.reset()
for i in range(14):
    action = []
    # print(env.step([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    for j in range(env.cars):
        action.append(random.random())
    print(env.step(action))

print(env.reward_history)


def something(b: BatteryDepreciationBase):
    b.calculate(5, 5, 5)


m = MyBatteryDepreciation()
something(m)
