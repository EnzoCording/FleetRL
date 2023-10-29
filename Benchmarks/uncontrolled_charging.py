import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from FleetRL.fleet_env.fleet_environment import FleetEnv

if __name__ == "__main__":

    # define parameters here for easier change
    n_steps = 8600
    n_episodes = 1
    n_evs = 1
    n_envs = 1
    timesteps_per_hour = 4
    use_case: str = "LMD"  # for file name

    dumb_vec_env = make_vec_env(FleetEnv,
                                    n_envs=n_envs,
                                    vec_env_cls=SubprocVecEnv,
                                    env_kwargs={
                                        "schedule_name": "lmd_sched_single_eval.csv",
                                        "building_name": "load_lmd.csv",
                                        "price_name": "spot_2021_new.csv",
                                        "tariff_name": "spot_2021_new_tariff.csv",
                                        "use_case": "lmd",
                                        "include_building": True,
                                        "include_pv": True,
                                        "time_picker": "static",
                                        "deg_emp": False,
                                        "include_price": True,
                                        "ignore_price_reward": False,
                                        "ignore_invalid_penalty": True,
                                        "ignore_overcharging_penalty": True,
                                        "ignore_overloading_penalty": False,
                                        "episode_length": n_steps,
                                        "normalize_in_env": False,
                                        "verbose": 0,
                                        "aux": True,
                                        "log_data": True,
                                        "calculate_degradation": True,
                                        "spot_markup": 10,
                                        "spot_mul": 1.5,
                                        "feed_in_ded": 0.25
                                    })
    # %%
    dumb_norm_vec_env = VecNormalize(venv=dumb_vec_env,
                                     norm_obs=True,
                                     norm_reward=True,
                                     training=True,
                                     clip_reward=10.0)

    episode_length = n_steps
    n_episodes = n_episodes
    dumb_norm_vec_env.reset()

    for i in range(episode_length * timesteps_per_hour * n_episodes):
        if dumb_norm_vec_env.env_method("is_done")[0]:
            dumb_norm_vec_env.reset()
        dumb_norm_vec_env.step([np.ones(n_evs)])

    dumb_log: pd.DataFrame = dumb_norm_vec_env.env_method("get_log")[0]

    dumb_log.reset_index(drop=True, inplace=True)
    dumb_log = dumb_log.iloc[0:-2]

    real_power_dumb = []
    for i in range(dumb_log.__len__()):
        dumb_log.loc[i, "hour_id"] = (dumb_log.loc[i, "Time"].hour + dumb_log.loc[i, "Time"].minute / 60)

    mean_per_hid_dumb = dumb_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
    mean_all_dumb = []
    for i in range(mean_per_hid_dumb.__len__()):
        mean_all_dumb.append(np.mean(mean_per_hid_dumb[i]))

    mean = pd.DataFrame()
    mean["Dumb charging"] = np.multiply(mean_all_dumb, 4)

    mean.plot()

    plt.xticks([0,8,16,24,32,40,48,56,64,72,80,88]
               ,["00:00","02:00","04:00","06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00"],
               rotation=45)

    plt.legend()
    plt.grid(alpha=0.2)

    plt.ylabel("Charging power in kW")
    max = dumb_log.loc[0, "Observation"][-10]
    plt.ylim([-max * 1.2, max * 1.2])

    plt.show()
