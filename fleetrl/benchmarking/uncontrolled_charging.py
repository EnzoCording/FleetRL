from fleetrl.fleet_env.fleet_environment import FleetEnv
from fleetrl.benchmarking.benchmark import Benchmark

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Uncontrolled(Benchmark):

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

        dumb_vec_env = make_vec_env(FleetEnv,
                                    env_kwargs=env_kwargs,
                                    n_envs=self.n_envs,
                                    vec_env_cls=SubprocVecEnv,
                                    seed=seed)

        dumb_norm_vec_env = VecNormalize(venv=dumb_vec_env,
                                         norm_obs=True,
                                         norm_reward=True,
                                         training=True,
                                         clip_reward=10.0)

        episode_length = self.n_steps
        n_episodes = self.n_episodes
        dumb_norm_vec_env.reset()

        self.env_config = env_kwargs["env_config"]

        for i in range(episode_length * self.time_steps_per_hour * n_episodes):
            if dumb_norm_vec_env.env_method("is_done")[0]:
                dumb_norm_vec_env.reset()
            dumb_norm_vec_env.step([np.ones(self.n_evs)])

        dumb_log: pd.DataFrame = dumb_norm_vec_env.env_method("get_log")[0]

        dumb_log.reset_index(drop=True, inplace=True)
        dumb_log = dumb_log.iloc[0:-2]

        return dumb_log

    def plot_benchmark(self,
                       dumb_log: pd.DataFrame,
                       ) -> None:

        dumb_log["hour_id"] = (dumb_log["Time"].dt.hour + dumb_log["Time"].dt.minute / 60)

        mean_per_hid_dumb = dumb_log.groupby("hour_id").mean()["Charging energy"].reset_index(drop=True)
        mean_all_dumb = []
        for i in range(mean_per_hid_dumb.__len__()):
            mean_all_dumb.append(np.mean(mean_per_hid_dumb[i]))

        mean = pd.DataFrame()
        mean["Dumb charging"] = np.multiply(mean_all_dumb, 4)

        mean.plot()

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
        max_val = dumb_log.loc[0, "Observation"][power_index]
        plt.ylim([-max_val * 1.2, max_val * 1.2])
        plt.show()
