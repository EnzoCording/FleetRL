from FleetRL.utils import data_processing
from FleetRL.utils import charge_ev
from FleetRL.utils import prices
from FleetRL.utils import load_calculation

import os
import random

import gym
import numpy as np
import pandas as pd


class FleetEnv(gym.Env):

    def __init__(self):

        # setting time-related model parameters
        self.freq = '15T'
        self.minutes = 15
        # self.freq = '1H'
        # self.minutes = 60
        self.hours = self.minutes / 60  # Hours per timestep, variable used in the energy calculations
        self.episode_length = 36  # episode length in hours
        self.end_cutoff = 2  # cutoff length at the end of the dataframe, in days. Used for choose_time
        self.price_window_size = 8  # number of hours look-ahead in price observation (day-ahead), max 12 hours

        if not (self.episode_length + self.price_window_size <= self.end_cutoff * 24):
            raise RuntimeError("Sum of episode length and price window size cannot exceed cutoff buffer.")

        # Setting EV parameters
        self.target_soc = 0.85  # Target SoC - Vehicles should always leave with this SoC
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.battery_cap = 65  # in kWh
        self.obc_max_power = 100  # onboard charger max power in kW
        self.charging_eff = 0.91  # charging efficiency
        self.discharging_eff = 0.91  # discharging efficiency

        # setting parameters of the company site
        self.company_case = "delivery"  # "delivery", "caretaker", "utility"
        # TODO: max power could change, I even have that info in the schedule
        # Grid connection: grid connection point max capacity in kW
        # EVSE (ev supply equipment aka charger) max power in kW
        self.grid_connection, self.evse_max_power = load_calculation.import_company(self.company_case)

        # initiating variables inside __init__()
        self.db = None
        self.time = None
        self.start_time = None
        self.finish_time = None
        self.soc = None
        self.next_soc = None
        self.hours_left = None
        self.price = None
        self.done = None
        self.info = {}
        self.reward_history = None
        self.cumulative_reward = None
        self.penalty_record = None
        self.max_time_left = None
        self.max_spot = None

        # rewards and penalties
        self.penalty_soc_violation = -5000
        self.penalty_overloading = -5000

        # initializing path name
        self.path_name = os.path.dirname(__file__) + '/../Input_Files/'

        self.db = data_processing.load_schedule(self)  # load schedule from defined pathname
        self.db = data_processing.compute_from_schedule(self)  # compute arriving SoC and time left for the trips

        # create a date range with the chosen frequency
        self.date_range = pd.DataFrame()
        self.date_range["date"] = pd.date_range(start=self.db["date"].min(),
                                                end=self.db["date"].max(),
                                                freq=self.freq
                                                )

        # Load spot price
        self.spot_price = prices.load_prices(self)

        # Get maximum values for normalization
        self.max_time_left = max(self.db["time_left"])
        self.max_spot = max(self.spot_price["DELU"])

        # Load building load and PV
        # TODO: implement these
        self.building_load = 35  # building load in kW
        self.pv = 0

        # first ID is 0
        # TODO: for now the number of cars is dictated by the data, but it could also be
        #  initialized in the class and then random EVs get picked from the database
        self.cars = self.db["ID"].max() + 1

        # setting the observation and action space of gym.Env
        # TODO: low 0 and high 1 because observations are normalised to [0, 1]
        # this makes the trained agent scalable
        '''
        >>> 2 * self.cars because of SOC and time_left
        >>> 24 * 3 is still unclear but currently PV, building load and price
        >>> for now 8 hours of price in to the future, always possible with DA
        '''

        # TODO: spot price updates during the day
        # TODO: how many points in the future should I give, do I need past values?
        # TODO: observation space has to always keep the same dimensions

        low_obs = np.array(np.zeros((2 * self.cars + self.price_window_size * (1.0/self.hours))), dtype=np.float32)
        high_obs = np.array(np.ones((2 * self.cars + self.price_window_size * (1.0/self.hours))), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.cars,), dtype=np.float32
        )

    def reset(self, start_time=None, **kwargs):
        # set done to False, since the episode just started
        self.done = False

        if not start_time:
            # choose a random start time
            self.start_time = self.choose_time()
        else:
            self.start_time = start_time

        # calculate the finish time based on the episode length
        self.finish_time = self.start_time + np.timedelta64(self.episode_length, 'h')

        # set the model time to the start time
        self.time = self.start_time

        obs = self.get_obs(self.time)

        # get the first soc and hours_left observation
        self.soc = obs[0]
        self.hours_left = obs[1]
        self.price = obs[2]

        ''' if time is insufficient due to unfavourable start date (for example loading an empty car with 15 min
        time left), soc is set in such a way that the agent always has a chance to fulfil the objective
        '''

        for car in range(self.cars):
            time_needed = ((self.target_soc - self.soc[car])
                           * self.battery_cap
                           / min([self.obc_max_power, self.evse_max_power]))

            # times 0.95 to give some tolerance, check if hours_left > 0: car has to be plugged in
            if (self.hours_left[car] > 0) and (time_needed > self.hours_left[car]):
                self.soc[car] = (self.target_soc
                                 - self.hours_left[car]
                                 * min([self.obc_max_power, self.evse_max_power]) * 0.95
                                 / self.battery_cap)
                print("Initial SOC modified due to unfavourable starting condition.")
                self.info["soc_mod"] = True

        # set the reward history back to an empty list, set cumulative reward to 0
        self.reward_history = []
        self.cumulative_reward = 0
        self.penalty_record = 0

        return self.normalize_obs([self.soc, self.hours_left, self.price]), self.info

    def step(self, action):  # , action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # TODO: Testing, trying to break it
        # TODO: comparing with chargym

        # parse the action to the charging function and receive the soc, next soc and reward
        self.soc, self.next_soc, reward = charge_ev.charge(self, action)

        if not load_calculation.check_violation(self, action):
            reward += self.penalty_overloading
            self.penalty_record += self.penalty_overloading
            print(f"Grid connection has been overloaded. "
                  f"Max possible: {self.grid_connection} kW, "
                  f"Actual: {sum(action) * self.evse_max_power + self.building_load} kW")

        # set the soc to the next soc
        self.soc = self.next_soc.copy()

        # advance one time step
        self.time += np.timedelta64(self.minutes, 'm')

        # get the next observation
        next_obs = self.get_obs(self.time)
        next_obs_soc = next_obs[0]
        next_obs_time_left = next_obs[1]
        next_obs_price = next_obs[2]

        self.price = next_obs_price

        # go through the cars and check whether the same car is still there, no car, or a new car
        for car in range(self.cars):

            # check if a car just left and didn't fully charge
            if (self.hours_left[car] != 0) and (next_obs_time_left[car] == 0):
                if self.target_soc - self.soc[car] > self.eps:
                    # TODO could scale with difference to SoC
                    # penalty for not fulfilling charging requirement
                    # TODO: this could be changed. No target SoC, but penalty if a car runs empty on a trip
                    reward += self.penalty_soc_violation
                    self.penalty_record += self.penalty_soc_violation
                    print("A car left the station without reaching the target SoC.")

            # same car in the next time step
            if (next_obs_time_left[car] != 0) and (self.hours_left[car] != 0):
                self.hours_left[car] -= self.hours

            # no car in the next time step
            elif next_obs_time_left[car] == 0:
                self.hours_left[car] = next_obs_time_left[car]
                self.soc[car] = next_obs_soc[car]

            # new car in the next time step
            elif (self.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                self.hours_left[car] = next_obs_time_left[car]
                self.soc[car] = next_obs_soc[car]

            # this shouldn't happen but if it does, an error is thrown
            else:
                raise TypeError("Observation format not recognized")

        # if the finish time is reached, set done to True
        # TODO: do I still experience the last timestep or do I finish when I reach it?
        # TODO: where is the environment reset?
        if self.time == self.finish_time:
            self.done = True

        # append to the reward history
        self.cumulative_reward += reward
        self.reward_history.append([self.time, self.cumulative_reward])

        # TODO: Here could be a saving function that saves the results of the episode

        # here, the reward is already in integer format
        return self.normalize_obs([self.soc, self.hours_left, self.price]), reward, self.done, self.info

    def close(self):
        pass

    def choose_time(self):
        # possible start times: remove last X days based on end_cutoff
        # TODO: same day can be pulled multiple times - is this a problem?
        possible_start_times = pd.date_range(start=self.db["date"].min(),
                                             end=(self.db["date"].max() - np.timedelta64(self.end_cutoff, 'D')),
                                             freq=self.freq
                                             )

        # choose a random start time and start the episode there
        chosen_start_time = random.choice(possible_start_times)

        # return start time
        return chosen_start_time

    def get_obs(self, time):
        # get the observation of soc and hours_left
        soc = self.db.loc[(self.db['date'] == time), 'SOC_on_return'].values
        hours_left = self.db.loc[(self.db['date'] == time), 'time_left'].values
        price_start = np.where(self.spot_price["date"] == self.time)[0][0]
        price_end = np.where(self.spot_price["date"] == (self.time + np.timedelta64(self.price_window_size,'h')))[0][0]
        price = self.spot_price["DELU"][price_start : price_end].values

        return [soc, hours_left, price]

    def normalize_obs(self, input_obs):
        input_obs[0] = input_obs[0]  # soc is already normalized
        input_obs[1] = input_obs[1] / self.max_time_left
        input_obs[2] = input_obs[2] / self.max_spot

        # TODO: concatenate everything into one np array

        return input_obs


    def render(self):
        # TODO: graph of rewards for example, or charging power or sth like that
        # TODO: Maybe a bar graph, centered at 0, n bars for n vehicles and height changes with power
        pass

    def construct_episode(self):
        # make a dataframe for the episode and only use that
        # date column, soc on arrival, time_left, building load, pv generation
        # can add phase angle etc. after computation
        # constraints: same month, same type of weekday, laxity >= 0

        pass
