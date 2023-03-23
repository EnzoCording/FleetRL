from FleetRL.utils import data_processing
from FleetRL.utils import charge_ev

import os
import random

import gym
import numpy as np
import pandas as pd


class FleetEnv(gym.Env):

    def __init__(self):

        # setting the frequency of the model
        self.freq = '15T'
        self.minutes = 15
        # self.freq = '1H'
        # self.minutes = 60
        self.hours = self.minutes / 60
        self.episode_length = 3  # episode length in hours
        self.end_cutoff = 2  # cutoff length at the end of the dataframe, in days. Used for choose_time

        # initializing self.db as None
        self.db = None

        # initializing path name
        self.path_name = os.path.dirname(__file__) + '/../Input_Files/full_test.csv'

        # EV parameters
        self.target_soc = 0.85  # Target SoC - Vehicles should always leave with this SoC
        self.battery_cap = 65  # in kWh
        self.obc_max_power = 100  # onboard charger max power in kW
        self.charging_eff = 0.91  # charging efficiency
        self.discharging_eff = 0.91  # discharging efficiency
        self.evse_max_power = 11  # EVSE (ev supply equipment) max power in kW

        self.db = data_processing.load_schedule(self)  # load schedule from defined pathname
        self.db = data_processing.compute_from_schedule(self)  # compute arriving SoC and time left for the trips

        # first ID is 0
        # TODO: for now the number of cars is dictated by the data, but it could also be
        #  initialized in the class and then random EVs get picked from the database
        self.cars = self.db["ID"].max() + 1

        # initiating variables inside __init__()
        self.time = None
        self.start_time = None
        self.finish_time = None
        self.soc = None
        self.next_soc = None
        self.hours_left = None
        self.done = None
        self.info = {}

    def reset(self):
        # set done to False, since the episode just started
        self.done = False
        # choose a random start time
        self.start_time = self.choose_time()
        # calculate the finish time based on the episode length
        self.finish_time = self.start_time + np.timedelta64(self.episode_length, 'h')
        # set the model time to the start time
        self.time = self.start_time
        # get the first soc and hours_left observation
        self.soc = self.get_obs(self.time)[0]
        self.hours_left = self.get_obs(self.time)[1]
        return [self.soc, self.hours_left], self.info

    def step(self, action):  # , action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # TODO: Testing, trying to break it
        # TODO: comparing with chargym

        # parse the action to the charging function and receive the soc, next soc and reward
        self.soc, self.next_soc, reward = charge_ev.charge(self, action)
        # set the soc to the next soc
        self.soc = self.next_soc.copy()
        # advance one time step
        self.time += np.timedelta64(self.minutes, 'm')
        # get the next observation
        next_obs_soc = self.get_obs(self.time)[0]
        next_obs_time_left = self.get_obs(self.time)[1]

        # go through the cars and check whether the same car is still there, no car, or a new car
        for car in range(self.cars):
            # same car
            if (next_obs_time_left[car] != 0) & (self.hours_left[car] != 0):
                self.hours_left[car] -= self.hours

            # no car
            elif next_obs_time_left[car] == 0:
                self.hours_left[car] = next_obs_time_left[car]
                self.soc[car] = next_obs_soc[car]

            # new car
            elif (next_obs_time_left[car] != 0) & (self.hours_left[car] == 0):
                self.hours_left[car] = next_obs_time_left[car]
                self.soc[car] = next_obs_soc[car]

            else:
                raise TypeError("Observation format not recognized")

        # if the finish time is reached, set done to True
        # TODO: do I still experience the last timestep or do I finish when I reach it?
        # TODO: where is the environment reset?
        if self.time == self.finish_time:
            self.done = True

        return [self.soc, self.hours_left], reward, self.done, self.info

    def close(self):
        pass

    def choose_time(self):
        # possible start times: remove last X days based on end_cutoff
        possible_start_times = pd.date_range(start=self.db["date"].min(),
                                             end=(self.db["date"].max() - np.timedelta64(self.end_cutoff, 'D')),
                                             freq=self.freq
                                             )

        # choose a random start time and start the episode there
        chosen_start_time = random.choice(possible_start_times)

        return chosen_start_time

    def get_obs(self, time):
        # get the observation of soc and hours_left
        soc = self.db.loc[(self.db['date'] == time), 'SOC_on_return'].values
        hours_left = self.db.loc[(self.db['date'] == time), 'time_left'].values

        return [soc, hours_left]
