import numpy as np
import pandas as pd

from fleetrl.fleet_env.config.time_config import TimeConfig


class Episode:

    """
    The Episode class holds all / most of the values that are episode-specific
    """

    def __init__(self, time_conf: TimeConfig):
        """
        Instantiating variables with default values.
        :param time_conf: Time config object
        """

        self.time_conf = time_conf  # time config object

        self.time: pd.Timestamp = None  # information of current time of the model
        self.start_time: pd.Timestamp = None  # starting date of the model (year needs to be the same as the schedule's)
        self.finish_time: pd.Timestamp = None  # ending date of the model (year needs to be the same as the schedule's)

        self.battery_cap: list = None  # battery capacity - changes with degradation
        self.soc: list = None  # State of charge of the battery
        self.soc_deg: list = None # State of charge for SoH calcs
        self.next_soc: list = None  # Next soc, information used in the step function
        self.next_soc_deg: list = None # Next State of charge for SoH calcs
        self.old_soc: list = None  # Previous soc, used to compute battery degradation
        self.old_soc_deg: list = None # Old State of charge for SoH calcs
        self.soh: np.ndarray = None  # state of health per car
        self.hours_left: list[float] = None  # Hours left at the charger
        self.price: list[float] = None  # Price in €/kWh
        self.done: bool = None  # Episode done or not
        self.truncated: bool = None  # Episode done due to time limit

        self.reward_history: list[tuple[pd.Timestamp, float]] = None  # History of reward development over the episode
        self.cumulative_reward: float = None  # Cumulative reward over an episode
        self.penalty_record: list[float] = None  # Record of penalty scores given

        self.current_charging_expense: float = None  # The amount spent per action
        self.total_charging_energy: float = None  # The amount of energy used per action
        self.log: pd.DataFrame = None  # DataFrame that logs what happens during the episode

        self.events: int = 0  # Variable that counts up if relevant events have been detected

        self.current_actions = None
