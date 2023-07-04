import pandas as pd
import numpy as np
import copy

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
                 charge_log: np.ndarray,
                 soh: np.ndarray):

        self.episode_count = len(self.log) // self.episode_length + 1

        self.entry = []

        self.entry.append({"Episode": copy.deepcopy(self.episode_count),
                           "Time": copy.deepcopy(time),
                           "Observation": copy.deepcopy(obs),
                           "Action": copy.deepcopy(action),
                           "Reward": copy.deepcopy(reward),
                           "Cashflow": copy.deepcopy(cashflow),
                           "Penalties": copy.deepcopy(penalty),
                           "Grid overloading": copy.deepcopy(grid),
                           "SOC violation": copy.deepcopy(soc_v),
                           "Degradation": copy.deepcopy(degradation),
                           "Charging energy": copy.deepcopy(charge_log),
                           "SOH": copy.deepcopy(soh)})

        self.log = pd.concat((self.log, pd.DataFrame(self.entry)), axis=0)
