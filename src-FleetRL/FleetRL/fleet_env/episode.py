import pandas as pd

from FleetRL.fleet_env.config.time_config import TimeConfig


class Episode:

    def __init__(self, time_conf: TimeConfig):

        self.time_conf = time_conf

        self.time: pd.Timestamp = None  # information of current time of the model
        self.start_time: pd.Timestamp = None
        self.finish_time: pd.Timestamp = None

        self.soc: list = None  # State of charge of the battery
        self.next_soc: list = None  # Next soc, information used in the step function
        self.old_soc: list = None  # Previous soc, used to compute battery degradation
        self.soh: list = None
        self.hours_left: list[float] = None  # Hours left at the charger
        self.price: list[float] = None  # Price in €/kWh
        self.done: bool = None  # Episode done or not

        self.reward_history: list[tuple[pd.Timestamp, float]] = None  # History of reward development over the episode
        self.cumulative_reward: float = None  # Cumulative reward over an episode
        self.penalty_record: list[float] = None  # Record of penalty scores given

        self.current_charging_expense: float = None  # The amount spent per action
        self.total_charging_energy: float = None  # The amount of energy used per action
        self.log: pd.DataFrame = None  # DataFrame that logs what happens during the episode

