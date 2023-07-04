import random
import matplotlib.pyplot as plt
import rainflow as rf
import numpy as np
from FleetRL.fleet_env.fleet_environment import FleetEnv
# from FleetRL.utils.prices import load_prices

env = FleetEnv(include_pv=True,
               schedule_name="5_ct.csv",
               include_building=True,
               include_price=True,
               normalize_in_env=False,
               aux=True,
               calculate_degradation=True,
               log_data=True,
               episode_length=72,
               time_picker="static",
               )

env.reset()
steps = 72*4
for i in range(steps):
    action = []
    # print(env.step([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    for j in range(env.num_cars):
        if env.episode.done:
            env.reset()
        action.append(-1)

    print(env.step(action))

print(env.data_logger.log.head())
print(env.data_logger.log.tail())
# print(env.episode.reward_history)
