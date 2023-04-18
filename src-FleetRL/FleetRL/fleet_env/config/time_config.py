import pandas as pd


class TimeConfig:
    def __init__(self):
        self.episode_length = 24  # episode length in hours
        self.end_cutoff = 2  # cutoff length at the end of the dataframe, in days. Used for choose_time
        self.price_lookahead = 8  # number of hours look-ahead in price observation (day-ahead), max 12 hours

        if not (self.episode_length + self.price_lookahead <= self.end_cutoff * 24):
            raise RuntimeError("Sum of episode length and price lookahead cannot exceed cutoff buffer, "
                               "otherwise the price lookahead would eventually be out of bounds.")

        self.freq: str = '1H'  # TODO describe
        self.minutes: int = 60  # TODO describe
        self.time_steps_per_hour: int = 1  # TODO describe
        self.dt: float = self.minutes / 60  # Hours per timestep, variable used in the energy calculations
