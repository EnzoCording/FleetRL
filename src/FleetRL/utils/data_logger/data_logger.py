import pandas as pd
import numpy as np

class DataLogger:
    def __init__(self, episode_length):
        self.log: pd.DataFrame = pd.DataFrame()
        self.entry = None
        self.episode_count: int = 1
        self.episode_length = episode_length
        # column names depending on the env state
        # obs - soc, time left, etc.

    def log_data(self, time: pd.Timestamp,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: float,
                 cashflow: float,
                 penalty: float,
                 grid: float,
                 soc_v: float,
                 degradation: float,
                 charge_log: np.ndarray):

        self.episode_count = len(self.log) // self.episode_length + 1

        self.entry = []

        self.entry.append({"Episode": self.episode_count,
                           "Time": time,
                           "Observation": obs,
                           "Action": action,
                           "Reward": reward,
                           "Cashflow": cashflow,
                           "Penalties": penalty,
                           "Grid overloading": grid,
                           "SOC violation": soc_v,
                           "Degradation": degradation,
                           "Charging energy": charge_log})

        self.log = pd.concat((self.log, pd.DataFrame(self.entry)), axis=0)
