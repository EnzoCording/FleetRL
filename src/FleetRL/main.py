import random
import matplotlib.pyplot as plt
import rainflow as rf
import numpy as np
from FleetRL.fleet_env.fleet_environment import FleetEnv
from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation
from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
# from FleetRL.utils.prices import load_prices

env = FleetEnv(include_pv=True,
               include_building=True,
               include_price=True,
               normalize_in_env=False,
               aux=True,
               calculate_degradation=True,
               log_data=True,
               episode_length=48,
               time_picker="random",
               )

env.reset()
steps = 48*4
for i in range(steps):
    action = []
    # print(env.step([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    for j in range(env.num_cars):
        if env.episode.done:
            env.reset()
        action.append(1)

    print(env.step(action))

print(env.data_logger.log.head())
print(env.data_logger.log.tail())
# print(env.episode.reward_history)

'''
soc_log = []
for j in range(env.num_cars):
    soc_log.append([env.data_logger.soc_log[i][j] for i in range(25)])
    plt.plot(soc_log[j])
plt.show()
for rng, mean, count, i_start, i_end in rf.extract_cycles(soc_log[0]):
    print(rng, mean, count, i_start, i_end)
'''
# soh_log = []
# soh_2 = []

# for j in range(env.num_cars):
#     soh_log.append([env.data_logger.soh_log[i][j] for i in range(steps)])
#     plt.plot(soh_log[j])
#     plt.legend(["SEI formation + Rainflow cycle counting", "Empirical linear degradation"])
# plt.show()

# %% Rainflow

import rainflow as rf
import numpy as np
import pandas as pd
