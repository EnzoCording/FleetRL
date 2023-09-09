import pandas as pd
import numpy as np
import copy

class DataLogger:
    """
    Logs data to allow for postprocessing, graphs, etc.
    The log is a dataframe, where each row can be a float or an array.
    Deepcopy to avoid risk of mutability (logs pointing back to changing variables)
    """
    def __init__(self, episode_length):
        """
        Initialising default values
        :param episode_length: Length of the episode in hours
        """
        self.log: pd.DataFrame = pd.DataFrame()
        self.entry = None
        self.episode_count: int = 1  # counter if several episodes are evaluated
        self.episode_length = episode_length

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
        """
        While the env object is the same, the episode counter will recognise different episodes.
        A dict is created with the required values, and then appended to the log dataframe

        :param time: Current timestamp
        :param obs: Observation array
        :param action: Action array
        :param reward: Reward float
        :param cashflow: in EUR
        :param penalty: penalty float
        :param grid: Grid connection in kW
        :param soc_v: Amount of SOC violated in # of batteries
        :param degradation: Degradation in that timestep
        :param charge_log: How much energy flowed into the batteries in kWh
        :param soh: Current SoH, array
        :return: None
        """

        self.episode_count = len(self.log) // self.episode_length + 1

        self.entry = []

        #  deepcopy to avoid mutable variables in log
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
