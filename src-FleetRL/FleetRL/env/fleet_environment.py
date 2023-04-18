import os

import gym
import numpy as np

from FleetRL.env.config.ev_config import EvConfig
from FleetRL.env.config.score_config import ScoreConfig
from FleetRL.env.config.time_config import TimeConfig
from FleetRL.env.episode import Episode
from FleetRL.utils.data_processing.data_processing import DataLoader
from FleetRL.utils.ev_charging.ev_charger import EvCharger
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation, CompanyType
from FleetRL.utils.normalization.normalization import Normalization
from FleetRL.utils.normalization.oracle_normalization import OracleNormalization
from FleetRL.utils.normalization.unit_normalization import UnitNormalization
from FleetRL.utils.observation.basic_observer import BasicObserver
from FleetRL.utils.observation.observer import Observer
from FleetRL.utils.time_picker.static_time_picker import StaticTimePicker
from FleetRL.utils.time_picker.time_picker import TimePicker


class FleetEnv(gym.Env):

    def __init__(self):

        # setting time-related model parameters
        # self.freq = '15T'
        # self.minutes = 15
        # self.time_steps_per_hour = 4
        self.ev_charger: EvCharger = EvCharger()  # class simulating EV charging
        self.time_picker: TimePicker = StaticTimePicker()  # when an episode starts, this class picks the starting time
        self.observer: Observer = BasicObserver()  # all observations are processed in the Observer class

        self.time_conf = TimeConfig()
        self.episode = Episode(self.time_conf)

        # Setting EV parameters
        self.target_soc = 0.85  # Target SoC - Vehicles should always leave with this SoC
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%

        self.ev_conf = EvConfig()

        self.load_calculation = LoadCalculation(CompanyType.Delivery)

        # initiating variables inside __init__()
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)

        self.max_time_left: float = None  # The maximum value of time left in the db dataframe
        self.max_spot: float = None  # The maximum spot market price in the db dataframe

        self.score_conf = ScoreConfig()

        # path for input files
        self.path_name = os.path.dirname(__file__) + '/../Input_Files/'

        # names of files to use
        # EV schedule database
        # self.file_name = 'full_test_one_car.csv'
        self.db_name = 'one_day_same_training.csv'
        # Spot price database
        self.spot_name = 'spot_2020.csv'

        # Loading thdatabase of EV schedules
        self.data_loader: DataLoader = DataLoader(self.path_name, self.db_name, self.spot_name, self.time_conf, self.ev_conf, self.target_soc)
        self.db = self.data_loader.db

        # Load spot price
        self.spot_price = self.data_loader.spot_price

        # Load building load and PV
        # TODO: implement these
        self.building_load = 35  # building load in kW
        self.pv = 0  # local PV rooftop production

        # first ID is 0
        # TODO: for now the number of cars is dictated by the data, but it could also be
        #  initialized in the class and then random EVs get picked from the database
        self.num_cars = self.db["ID"].max() + 1

        # TODO: spot price updates during the day, to allow more than 8 hour lookahead at some times
        #  (use clipping if not available, repeat last value in window)
        # TODO: how many points in the future should I give, do I need past values? probably not
        # Remember: observation space has to always keep the same dimensions

        self.normalizer: Normalization = OracleNormalization(self.db, self.spot_price)
        # self.normalizer: Normalization = \
        #     UnitNormalization(self.db, self.spot_price, self.num_cars,
        #                       self.time_conf.price_lookahead, self.time_conf.time_steps_per_hour)

        low_obs, high_obs = self.normalizer.make_boundaries(
            (2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour)
        )

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cars,), dtype=np.float32)

    def reset(self, start_time=None, **kwargs):
        # set done to False, since the episode just started
        self.episode.done = False

        if not start_time:
            # choose a random start time
            self.episode.start_time = self.time_picker.choose_time(self.db, self.time_conf.freq,
                                                                   self.time_conf.end_cutoff)
        else:
            self.episode.start_time = start_time

        # calculate the finish time based on the episode length
        self.episode.finish_time = self.episode.start_time + np.timedelta64(self.time_conf.episode_length, 'h')

        # set the model time to the start time
        self.episode.time = self.episode.start_time

        obs = self.observer.get_obs(self.db, self.spot_price, self.time_conf.price_lookahead, self.episode.time)

        # get the first soc and hours_left observation
        self.episode.soc = obs[0]
        self.episode.hours_left = obs[1]
        self.episode.price = obs[2]

        ''' if time is insufficient due to unfavourable start date (for example loading an empty car with 15 min
        time left), soc is set in such a way that the agent always has a chance to fulfil the objective
        '''

        for car in range(self.num_cars):
            time_needed = ((self.target_soc - self.episode.soc[car])
                           * self.ev_conf.battery_cap
                           / min([self.ev_conf.obc_max_power, self.load_calculation.evse_max_power]))

            # times 0.8 to give some tolerance, check if hours_left > 0: car has to be plugged in
            if (self.episode.hours_left[car] > 0) and (time_needed > self.episode.hours_left[car]):
                self.episode.soc[car] = (self.target_soc
                                 - self.episode.hours_left[car]
                                 * min([self.ev_conf.obc_max_power, self.load_calculation.evse_max_power]) * 0.8
                                 / self.ev_conf.battery_cap)
                print("Initial SOC modified due to unfavourable starting condition.")

        # set the reward history back to an empty list, set cumulative reward to 0
        self.episode.reward_history = []
        self.episode.cumulative_reward = 0
        self.episode.penalty_record = 0

        return self.normalizer.normalize_obs([self.episode.soc, self.episode.hours_left, self.episode.price])

    def step(self, actions):  # , action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # TODO: Testing, trying to break it
        # TODO: comparing with chargym

        # parse the action to the charging function and receive the soc, next soc and reward
        self.episode.soc, self.episode.next_soc, reward = self.ev_charger.charge(
            self.db, self.spot_price, self.num_cars, actions, self.episode, self.load_calculation,
            self.ev_conf,  self.time_conf, self.score_conf)

        # at this point, the reward only includes the current expense/revenue of the charging process
        self.episode.current_charging_expense = reward

        # calling the print function
        self.print(actions)

        # print(f"Max possible: {self.grid_connection} kW" ,f"Actual: {sum(action) * self.evse_max_power + self.building_load} kW")
        # print(self.price[0])
        # print(self.hours_left)

        # check if the current load exceeds the trafo rating and penalize
        if not self.load_calculation.check_violation(self.building_load, actions, self.pv):
            reward += self.score_conf.penalty_overloading
            self.episode.penalty_record += self.score_conf.penalty_overloading
            print(f"Grid connection has been overloaded. "
                  f"Max possible: {self.load_calculation.grid_connection} kW, "
                  f"Actual: {sum(actions) * self.load_calculation.evse_max_power + self.building_load} kW")

        # set the soc to the next soc
        self.episode.soc = self.episode.next_soc.copy()

        # advance one time step
        self.episode.time += np.timedelta64(self.time_conf.minutes, 'm')

        # get the next observation
        next_obs = self.observer.get_obs(self.db, self.spot_price, self.time_conf.price_lookahead, self.episode.time)
        next_obs_soc = next_obs[0]
        next_obs_time_left = next_obs[1]
        next_obs_price = next_obs[2]

        self.episode.price = next_obs_price

        # go through the cars and check whether the same car is still there, no car, or a new car
        for car in range(self.num_cars):

            # check if a car just left and didn't fully charge
            if (self.episode.hours_left[car] != 0) and (next_obs_time_left[car] == 0):
                if self.target_soc - self.episode.soc[car] > self.eps:
                    # TODO could scale with difference to SoC
                    # penalty for not fulfilling charging requirement
                    # TODO: this could be changed. No target SoC, but penalty if a car runs empty on a trip
                    reward += self.score_conf.penalty_soc_violation
                    self.episode.penalty_record += self.score_conf.penalty_soc_violation
                    print("A car left the station without reaching the target SoC.")

            # same car in the next time step
            if (next_obs_time_left[car] != 0) and (self.episode.hours_left[car] != 0):
                self.episode.hours_left[car] -= self.time_conf.dt

            # no car in the next time step
            elif next_obs_time_left[car] == 0:
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.soc[car] = next_obs_soc[car]

            # new car in the next time step
            elif (self.episode.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.soc[car] = next_obs_soc[car]

            # this shouldn't happen but if it does, an error is thrown
            else:
                raise TypeError("Observation format not recognized")

        # if the finish time is reached, set done to True
        # TODO: do I still experience the last timestep or do I finish when I reach it?
        # TODO: where is the environment reset?
        if self.episode.time == self.episode.finish_time:
            self.episode.done = True
            print(f"Episode done: {self.episode.done}")

        # append to the reward history
        self.episode.cumulative_reward += reward
        self.episode.reward_history.append((self.episode.time, self.episode.cumulative_reward))

        # TODO: Here could be a saving function that saves the results of the episode

        print(f"Reward signal: {reward}")
        print("---------")
        print("\n")

        # here, the reward is already in integer format
        return (self.normalizer.normalize_obs([self.episode.soc, self.episode.hours_left, self.episode.price]),
                reward, self.episode.done, self.info)

    def close(self):
        return 0

    def print(self, action):
        print(f"Timestep: {self.episode.time}")
        print(f"Price: {self.episode.price[0] / 1000} €/kWh")
        print(f"SOC: {self.episode.soc}, Time left: {self.episode.hours_left} hours")
        print(f"Action taken: {action}")
        print(f"Actual charging energy: {self.episode.total_charging_energy} kWh")
        print(f"Charging cost/revenue: {self.episode.current_charging_expense} €")
        print("--------------------------")

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