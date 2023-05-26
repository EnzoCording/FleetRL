import random
import matplotlib.pyplot as plt
import rainflow as rf

from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation
from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
# from FleetRL.utils.prices import load_prices

env = FleetEnv()
env.reset()
for i in range(1500):
    action = []
    # print(env.step([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    for j in range(env.num_cars):
        action.append(random.uniform(-1,1))
    print(env.step(action))

print(env.episode.reward_history)
soc_log = []
for j in range(env.num_cars):
    soc_log.append([env.data_logger.soc_log[i][j] for i in range(25)])
    plt.plot(soc_log[j])
plt.show()
for rng, mean, count, i_start, i_end in rf.extract_cycles(soc_log[0]):
    print(rng, mean, count, i_start, i_end)

