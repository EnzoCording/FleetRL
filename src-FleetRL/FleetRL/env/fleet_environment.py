import os

import gym
import numpy as np
import pandas as pd

from FleetRL.utils import data_processing, prices
from FleetRL.utils.ev_charging.ev_charger import EvCharger
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.utils.normalization.normalization import Normalization
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

        self.observer: Observer = BasicObserver()

        self.freq: str = '1H'  # TODO describe
        self.minutes: int = 60  # TODO describe
        self.time_steps_per_hour: int = 1  # TODO describe
        self.hours: float = self.minutes / 60  # Hours per timestep, variable used in the energy calculations

        self.episode_length = 24  # episode length in hours
        self.end_cutoff = 2  # cutoff length at the end of the dataframe, in days. Used for choose_time
        self.price_lookahead = 8  # number of hours look-ahead in price observation (day-ahead), max 12 hours

        if not (self.episode_length + self.price_lookahead <= self.end_cutoff * 24):
            raise RuntimeError("Sum of episode length and price window size cannot exceed cutoff buffer.")

        # Setting EV parameters
        self.target_soc = 0.85  # Target SoC - Vehicles should always leave with this SoC
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.battery_cap = 65  # battery capacity in kWh
        self.obc_max_power = 100  # onboard charger max power in kW
        self.charging_eff = 0.91  # charging efficiency
        self.discharging_eff = 0.91  # discharging efficiency

        self.load_calculation = LoadCalculation("delivery")

        # initiating variables inside __init__()
        self.db: pd.DataFrame = None  # database of EV schedules
        self.time: pd.Timestamp = None  # information of current time
        self.start_time: pd.Timestamp = None
        self.finish_time: pd.Timestamp = None
        self.soc: list = None  # State of charge of the battery
        self.next_soc: list = None  # Next soc, information used in the step function
        self.hours_left: list[float] = None  # Hours left at the charger
        self.price: list[float] = None  # Price in €/kWh
        self.done: bool = None  # Episode done or not
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)
        self.reward_history: list[tuple[pd.Timestamp, float]] = None  # History of reward development over the episode
        self.cumulative_reward: float = None  # Cumulative reward over an episode
        self.penalty_record: list[float] = None  # Record of penalty scores given
        self.max_time_left: float = None  # The maximum value of time left in the db dataframe
        self.max_spot: float = None  # The maximum spot market price in the db dataframe
        self.current_charging_expense: float = None  # The amount spent per action
        self.total_charging_energy: float = None  # The amount of energy used per action

        # penalties for violations
        self.penalty_soc_violation = -500_000.0
        self.penalty_overloading = -50_000.0
        self.penalty_invalid_action = -1

        # possible reward: money spent/earned due to buying/selling electricity for charging/discharging

        # path for input files
        self.path_name = os.path.dirname(__file__) + '/../Input_Files/'

        # names of files to use
        # EV schedule database
        # self.file_name = 'full_test_one_car.csv'
        self.db_name = 'one_day_same_training.csv'
        # Spot price database
        self.spot_name = 'spot_2020.csv'

        self.db = data_processing.load_schedule(self)  # load schedule from defined pathname
        self.db = data_processing.compute_from_schedule(self)  # compute arriving SoC and time left for the trips

        # create a date range with the chosen frequency
        # Given the desired frequency, create a (down-sampled) column of timestamps
        self.date_range = pd.DataFrame()
        self.date_range["date"] = pd.date_range(start=self.db["date"].min(),
                                                end=self.db["date"].max(),
                                                freq=self.freq
                                                )

        # Load spot price
        self.spot_price = prices.load_prices(self)



        # Load building load and PV
        # TODO: implement these
        self.building_load = 35  # building load in kW
        self.pv = 0  # local PV rooftop production

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

        # TODO: spot price updates during the day, to allow more than 8 hour lookahead at some times
        #  (use clipping if not available, repeat last value in window)
        # TODO: how many points in the future should I give, do I need past values? probably not
        # Remember: observation space has to always keep the same dimensions

        # TODO move this into observation, these bounds depend on observation strategy

        # self.normalizer: Normalization = OracleNormalization(self.db, self.spot_price)
        self.normalizer: Normalization = \
            UnitNormalization(self.db, self.spot_price, self.cars,
                                            self.price_lookahead, self.time_steps_per_hour)

        low_obs, high_obs = self.normalizer.make_boundaries(
            (2 * self.cars + self.price_lookahead * self.time_steps_per_hour)
        )

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.cars,), dtype=np.float32)

    def reset(self, start_time=None, **kwargs):
        # set done to False, since the episode just started
        self.done = False

        if not start_time:
            # choose a random start time
            self.start_time = self.time_picker.choose_time(self.db, self.freq, self.end_cutoff)
        else:
            self.start_time = start_time

        # calculate the finish time based on the episode length
        self.finish_time = self.start_time + np.timedelta64(self.episode_length, 'h')

        # set the model time to the start time
        self.time = self.start_time

        obs = self.observer.get_obs(self.db, self.spot_price, self.price_lookahead, self.time)

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
                           / min([self.obc_max_power, self.load_calculation.evse_max_power]))

            # times 0.8 to give some tolerance, check if hours_left > 0: car has to be plugged in
            if (self.hours_left[car] > 0) and (time_needed > self.hours_left[car]):
                self.soc[car] = (self.target_soc
                                 - self.hours_left[car]
                                 * min([self.obc_max_power, self.load_calculation.evse_max_power]) * 0.8
                                 / self.battery_cap)
                print("Initial SOC modified due to unfavourable starting condition.")

        # set the reward history back to an empty list, set cumulative reward to 0
        self.reward_history = []
        self.cumulative_reward = 0
        self.penalty_record = 0

        return self.normalizer.normalize_obs([self.soc, self.hours_left, self.price])

    def step(self, actions):  # , action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # TODO: Testing, trying to break it
        # TODO: comparing with chargym

        # parse the action to the charging function and receive the soc, next soc and reward
        self.soc, self.next_soc, reward = self.ev_charger.charge(
            self.db, self.time, self.hours, self.battery_cap, self.charging_eff, self.discharging_eff,
            self.spot_price, self.soc, self.cars, actions, self.obc_max_power,
            self.load_calculation.evse_max_power, self.penalty_invalid_action
        )   # TODO move some fields into ev_charger, since it should have ownership over e.g.

        # at this point, the reward only includes the current expense/revenue of the charging process
        self.current_charging_expense = reward

        self.print(actions)

        # print(f"Max possible: {self.grid_connection} kW" ,f"Actual: {sum(action) * self.evse_max_power + self.building_load} kW")
        # print(self.price[0])
        # print(self.hours_left)

        if not self.load_calculation.check_violation(self.building_load, actions, self.pv):
            reward += self.penalty_overloading
            self.penalty_record += self.penalty_overloading
            print(f"Grid connection has been overloaded. "
                  f"Max possible: {self.load_calculation.grid_connection} kW, "
                  f"Actual: {sum(actions) * self.load_calculation.evse_max_power + self.building_load} kW")

        # set the soc to the next soc
        self.soc = self.next_soc.copy()

        # advance one time step
        self.time += np.timedelta64(self.minutes, 'm')

        # get the next observation
        next_obs = self.observer.get_obs(self.db, self.spot_price, self.price_lookahead, self.time)
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
            print(f"Episode done: {self.done}")

        # append to the reward history
        self.cumulative_reward += reward
        self.reward_history.append((self.time, self.cumulative_reward))

        # TODO: Here could be a saving function that saves the results of the episode

        print(f"Reward signal: {reward}")
        print("---------")
        print("\n")

        # here, the reward is already in integer format
        return self.normalizer.normalize_obs([self.soc, self.hours_left, self.price]), reward, self.done, self.info

    def close(self):
        return 0

    def print(self, action):
        print(f"Timestep: {self.time}")
        print(f"Price: {self.price[0] / 1000} €/kWh")
        print(f"SOC: {self.soc}, Time left: {self.hours_left} hours")
        print(f"Action taken: {action}")
        print(f"Actual charging energy: {self.total_charging_energy} kWh")
        print(f"Charging cost/revenue: {self.current_charging_expense} €")
        print("--------------------------")
        print("\n")

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
