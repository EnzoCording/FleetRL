import pandas as pd


class TimeConfig:
    def __init__(self):
        self.episode_length = 24  # episode length in hours
        self.end_cutoff = 6  # cutoff length at the end of the dataframe, in days. Used for choose_time
        self.price_lookahead = 8  # number of hours look-ahead in price observation (day-ahead), max 12 hours
        self.bl_pv_lookahead = 4  # look-ahead in load and pv, using future values, should be forecasted

        # TODO: needs to be dealt with using dates
        '''
        if not (self.episode_length + self.price_lookahead <= self.end_cutoff * 24):
            raise RuntimeError("Sum of episode length and price lookahead cannot exceed cutoff buffer, "
                               "otherwise the price lookahead would eventually be out of bounds.")
        '''

        # setting time-related model parameters
        # self.freq = '1H'
        # self.minutes = '60'
        # self.time_steps_per_hour = 1

        # NB: when using hourly frequency, some info can get lost, causing inaccuracies

        self.freq: str = '15T'  # Frequency string needed to down-sample in pandas
        self.minutes: int = 15  # Amount of minutes per time step
        self.time_steps_per_hour: int = 4  # Number of time steps per hour, used in obs_space
        self.dt: float = self.minutes / 60  # Hours per timestep, variable used in the energy calculations
