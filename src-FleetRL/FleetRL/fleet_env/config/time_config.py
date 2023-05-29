import pandas as pd


class TimeConfig:
    def __init__(self):
        self.episode_length = 24  # episode length in hours
        self.end_cutoff = 2  # cutoff length at the end of the dataframe, in days. Used for choose_time
        self.price_lookahead = 8  # number of hours look-ahead in price observation (day-ahead), max 12 hours

        # TODO: needs to be dealt with using dates
        '''
        if not (self.episode_length + self.price_lookahead <= self.end_cutoff * 24):
            raise RuntimeError("Sum of episode length and price lookahead cannot exceed cutoff buffer, "
                               "otherwise the price lookahead would eventually be out of bounds.")
        '''

        # setting time-related model parameters
        # self.freq = '15T'
        # self.minutes = 15
        # self.time_steps_per_hour = 4

        self.freq: str = '1H'  # Frequency string needed to down-sample in pandas
        self.minutes: int = 60  # Amount of minutes per time step
        self.time_steps_per_hour: int = 1  # Number of time steps per hour, used in obs_space
        self.dt: float = self.minutes / 60  # Hours per timestep, variable used in the energy calculations