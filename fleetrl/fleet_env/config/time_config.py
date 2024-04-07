class TimeConfig:

    """
    The Time config sets parameters for the episode and MDP dynamics.
    - end_cutoff: Default at 60. The last two months are not regarding in the training observations
    - The last two months are used as a validation set during training
    """

    def __init__(self, env_config):
        self.episode_length = env_config.get('episode_length', 24)  # episode length in hours
        self.end_cutoff = env_config.get('end_cutoff', 60)  # cutoff length at the end of the dataframe, in days
        self.price_lookahead = env_config.get('price_lookahead', 8)  # hours look-ahead in price observation, max 12 hrs
        self.bl_pv_lookahead = env_config.get('bl_pv_lookahead', 4)  # look-ahead in load and pv, using future values

        # setting time-related model parameters
        # self.freq = '1H'
        # self.minutes = '60'
        # self.time_steps_per_hour = 1
        # NB: when using hourly frequency, some info can get lost, causing inaccuracies (down-sampling)

        self.freq = env_config.get('freq', '15T')  # Frequency string needed to sample in pandas
        self.minutes = env_config.get('minutes', 15)  # Amount of minutes per time step
        self.time_steps_per_hour = env_config.get('time_steps_per_hour', 4)  # Number of time steps per hour
        self.dt: float = self.minutes / 60  # Hours per timestep, variable used in the energy calculations
