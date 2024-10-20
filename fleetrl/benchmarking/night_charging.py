import math
from copy import copy

from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NightCharging(Benchmark):

    def __init__(self,
                 n_steps: int,
                 n_evs: int,
                 n_episodes: int = 1,
                 n_envs: int = 1,
                 time_steps_per_hour: int = 4):

        self.n_steps = n_steps
        self.n_evs = n_evs
        self.n_episodes = n_episodes
        self.n_envs = n_envs
        self.time_steps_per_hour = time_steps_per_hour
        self.env_config = None

    def run_benchmark(self,
                      use_case: str,
                      env_kwargs: dict,
                      seed: int = None
                      ) -> pd.DataFrame:

        night_vec_env = make_vec_env(FleetEnv,
                                     n_envs=self.n_envs,
                                     vec_env_cls=SubprocVecEnv,
                                     env_kwargs=env_kwargs,
                                     seed=seed)

        night_norm_vec_env = VecNormalize(venv=night_vec_env,
                                          norm_obs=True,
                                          norm_reward=True,
                                          training=True,
                                          clip_reward=10.0)

        env_config = env_kwargs["env_config"]

        env = FleetEnv(env_config)

        df = env.db
        df_leaving_home = df[(df['Location'].shift() == 'home') & (df['Location'] == 'driving')]
        earliest_dep_time = df_leaving_home['date'].dt.time.min()
        day_of_earliest_dep = df_leaving_home[df_leaving_home['date'].dt.time == earliest_dep_time]['date'].min()
        earliest_dep = earliest_dep_time.hour + earliest_dep_time.minute / 60

        evse = env.load_calculation.evse_max_power
        cap = env.ev_config.init_battery_cap
        target_soc = env.ev_config.target_soc
        eff = env.ev_config.charging_eff

        max_time_needed = target_soc * cap / eff / evse  # time needed to charge to target soc from 0
        difference = earliest_dep - max_time_needed
        starting_time = (24 + difference)
        if starting_time > 24:
            starting_time = 23.99  # always start just before midnight

        charging_hour = int(math.modf(starting_time)[1])
        minutes = np.asarray([0, 15, 30, 45])
        # split number and decimals, use decimals and choose the closest minute
        closest_index = np.abs(minutes - int(math.modf(starting_time)[0] * 60)).argmin()
        charging_minute = minutes[closest_index]

        episode_length = self.n_steps
        n_episodes = self.n_episodes
        night_norm_vec_env.reset()

        charging = False

        for i in range(episode_length * self.time_steps_per_hour * n_episodes):
            if night_norm_vec_env.env_method("is_done")[0]:
                night_norm_vec_env.reset()
            time: pd.Timestamp = night_norm_vec_env.env_method("get_time")[0]
            if ((time.hour >= 11) and (time.hour <= 14)) and (use_case == "ct"):
                night_norm_vec_env.step(
                    ([np.clip(np.multiply(np.ones(self.n_evs), night_norm_vec_env.env_method("get_dist_factor")[0]), 0, 1)]))
                continue
            time: pd.Timestamp = night_norm_vec_env.env_method("get_time")[0]
            if (((charging_hour <= time.hour) and (charging_minute <= time.minute)) or (charging)):
                if not charging:
                    charging_start: pd.Timestamp = copy(time)
                charging = True
                night_norm_vec_env.step([np.ones(self.n_evs)])
            else:
                night_norm_vec_env.step([np.zeros(self.n_evs)])
            if charging and ((time - charging_start).total_seconds() / 3600 > int(max_time_needed)):
                charging = False

        night_log: pd.DataFrame = night_norm_vec_env.env_method("get_log")[0]

        night_log.reset_index(drop=True, inplace=True)
        night_log = night_log.iloc[0:-2]

        self.env_config = env_kwargs["env_config"]

        return night_log

    def plot_benchmark(self,
                       night_log: pd.DataFrame,
                       ) -> None:

        night_log["hour_id"] = (night_log["Time"].dt.hour + night_log["Time"].dt.minute / 60)

        mean_per_hid_night = night_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
        mean_all_night = []
        for i in range(mean_per_hid_night.__len__()):
            mean_all_night.append(np.mean(mean_per_hid_night[i]))

        mean_night = pd.DataFrame()
        mean_night["Night charging"] = np.multiply(mean_all_night, 4)

        mean_night.plot()

        plt.xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
                   , ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00",
                      "22:00"],
                   rotation=45)

        plt.legend()
        plt.grid(alpha=0.2)

        plt.ylabel("Charging power in kW")
        price_lookahead = self.env_config["price_lookahead"] * int(self.env_config["include_price"])
        bl_pv_lookahead = self.env_config["bl_pv_lookahead"]
        number_of_lookaheads = sum([int(self.env_config["include_pv"]), int(self.env_config["include_building"])])
        # check observer module for building of observation list
        power_index = self.n_evs * 6 + 2 * (price_lookahead+1) + number_of_lookaheads * (bl_pv_lookahead+1) + 1
        max_val = night_log.loc[0, "Observation"][power_index]
        plt.ylim([-max_val * 1.2, max_val * 1.2])
        plt.show()
