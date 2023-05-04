import os

import gymnasium as gym
import numpy as np

from FleetRL.fleet_env.config.ev_config import EvConfig
from FleetRL.fleet_env.config.score_config import ScoreConfig
from FleetRL.fleet_env.config.time_config import TimeConfig

from FleetRL.fleet_env.episode import Episode

from FleetRL.utils.data_processing.data_processing import DataLoader
from FleetRL.utils.ev_charging.ev_charger import EvCharger
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation, CompanyType

from FleetRL.utils.normalization.normalization import Normalization
from FleetRL.utils.normalization.oracle_normalization import OracleNormalization
from FleetRL.utils.normalization.unit_normalization import UnitNormalization

from FleetRL.utils.observation.observer_with_building_load import ObserverWithBuildingLoad
from FleetRL.utils.observation.basic_observer import BasicObserver
from FleetRL.utils.observation.observer import Observer

from FleetRL.utils.time_picker.random_time_picker import RandomTimePicker
from FleetRL.utils.time_picker.static_time_picker import StaticTimePicker
from FleetRL.utils.time_picker.time_picker import TimePicker

from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation


class FleetEnv(gym.Env):

    def __init__(self):

        # Setting paths and file names
        # path for input files, needs to be the same for all inputs
        self.path_name = os.path.dirname(__file__) + '/../Input_Files/'
        # EV schedule database
        self.schedule_name = 'full_test_one_car.csv'
        # self.schedule_name = 'one_day_same_training.csv'
        # Spot price database
        self.spot_name = 'spot_2020.csv'
        # Building load database
        self.building_name = 'caretaker.csv'
        # PV gen database
        self.pv_name = None

        # Setting flags for the type of environment to build
        # NOTE: they are appended to the db in the order specified here
        # NOTE: import the right observer!
        self.include_building_load = True
        self.include_pv = False

        # Loading configs
        self.time_conf = TimeConfig()
        self.ev_conf = EvConfig()
        self.score_conf = ScoreConfig()
        self.load_calculation = LoadCalculation(CompanyType.Delivery)

        # Loading modules
        self.ev_charger: EvCharger = EvCharger()  # class simulating EV charging
        self.time_picker: TimePicker = RandomTimePicker()  # when an episode starts, this class picks the starting time
        self.observer: Observer = ObserverWithBuildingLoad()  # all observations are processed in the Observer class
        self.episode: Episode = Episode(self.time_conf)
        self.battery_degradation: BatteryDegradation = EmpiricalDegradation()

        # Setting EV parameters
        self.target_soc = 0.85  # Target SoC - Vehicles should always leave with this SoC
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.initial_soh = 1.0  # initial degree of battery degradation, assumed equal for all cars

        # initiating variables inside __init__() that are needed for gym.Env
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)
        self.max_time_left: float = None  # The maximum value of time left in the db dataframe
        self.max_spot: float = None  # The maximum spot market price in the db dataframe

        # Loading the inputs
        self.data_loader: DataLoader = DataLoader(self.path_name, self.schedule_name,
                                                  self.spot_name, self.building_name, self.pv_name,
                                                  self.time_conf, self.ev_conf, self.target_soc,
                                                  self.include_building_load, self.include_pv
                                                  )
        # # Get schedule
        # self.schedule = self.data_loader.schedule
        #
        # # Get spot price
        # self.spot_price = self.data_loader.spot_price
        #
        # # Get building load
        # self.building_load = self.data_loader.building_load
        #
        # get the total database
        self.db = self.data_loader.db

        # Load PV
        # TODO: implement this
        # self.pv = 0  # local PV rooftop production

        # first ID is 0
        # TODO: for now the number of cars is dictated by the data, but it could also be
        #  initialized in the class and then random EVs get picked from the database
        self.num_cars = self.db["ID"].max() + 1

        self.episode.soh = np.ones(self.num_cars) * self.initial_soh  # initialize soh for each battery

        # TODO: spot price updates during the day, to allow more than 8 hour lookahead at some times
        #  (use clipping if not available, repeat last value in window)
        # TODO: how many points in the future should I give, do I need past values? probably not
        # Remember: observation space has to always keep the same dimensions

        # Load gym observation spaces, decided which normalization strategy to choose
        self.normalizer: Normalization = OracleNormalization(self.db, self.include_building_load, self.include_pv)
        # unit normalization: doesn't normalize, only concatenates
        # self.normalizer: Normalization = \
        #     UnitNormalization(self.db, self.spot_price, self.num_cars,
        #                       self.time_conf.price_lookahead, self.time_conf.time_steps_per_hour)

        # set boundaries of the observation space, detects if normalized or not

        if not self.include_building_load and not self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour
            low_obs, high_obs = self.normalizer.make_boundaries(dim)
        elif self.include_building_load and not self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour + 1
            low_obs, high_obs = self.normalizer.make_boundaries(dim)
        else:
            low_obs = None
            high_obs = None
            raise RuntimeError("Problem with components. Check building and pv flags.")

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cars,), dtype=np.float32)

    def reset(self, **kwargs):
        # set done to False, since the episode just started
        self.episode.done = False

        self.episode.soh = np.ones(self.num_cars) * self.initial_soh

        self.episode.start_time = self.time_picker.choose_time(self.db, self.time_conf.freq,
                                                               self.time_conf.end_cutoff
                                                               )

        # calculate the finish time based on the episode length
        self.episode.finish_time = self.episode.start_time + np.timedelta64(self.time_conf.episode_length, 'h')

        # set the model time to the start time
        self.episode.time = self.episode.start_time

        obs = self.observer.get_obs(self.db, self.time_conf.price_lookahead, self.episode.time)

        # get the first soc and hours_left observation
        self.episode.soc = obs[0] * self.episode.soh
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
                                         * min([self.ev_conf.obc_max_power, self.load_calculation.evse_max_power])
                                         * 0.8 / self.ev_conf.battery_cap
                                         )
                print("Initial SOC modified due to unfavourable starting condition.")

        # set the reward history back to an empty list, set cumulative reward to 0
        self.episode.reward_history = []
        self.episode.cumulative_reward = 0
        self.episode.penalty_record = 0

        obs[0] = self.episode.soc
        obs[1] = self.episode.hours_left
        obs[2] = self.episode.price

        # TODO Unit normalizer boundaries
        return self.normalizer.normalize_obs(obs), self.info

    def step(self, actions):  # , action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # parse the action to the charging function and receive the soc, next soc, reward and cashflow
        # the soh is taken into account within the charge function
        self.episode.soc, self.episode.next_soc, reward, cashflow = self.ev_charger.charge(
            self.db, self.num_cars, actions, self.episode, self.load_calculation,
            self.ev_conf, self.time_conf, self.score_conf)

        # save the old soc for logging purposes
        self.episode.old_soc = self.episode.soc

        # at this point, the reward only includes the current expense/revenue of the charging process
        # not true anymore, violation of invalid charging
        # TODO: decouple reward and economic metrics
        # todo remove this
        self.episode.current_charging_expense = reward

        # calling the print function
        self.print(actions)

        # print(f"Max possible: {self.grid_connection} kW" ,f"Actual: {sum(action) * self.evse_max_power + self.building_load} kW")
        # print(self.price[0])
        # print(self.hours_left)

        # check if the current load exceeds the trafo rating and penalize accordingly

        if self.include_building_load:
            current_load = self.db.loc[self.db["date"] == self.episode.time, "load"].values
        else:
            current_load = 0

        if self.include_pv:
            current_pv = self.db[self.db["date"] == self.episode.time, "pv"].values
        else:
            current_pv = 0

        if not self.load_calculation.check_violation(current_load, actions, current_pv):
            reward += self.score_conf.penalty_overloading
            self.episode.penalty_record += self.score_conf.penalty_overloading
            print(f"Grid connection has been overloaded.")

        # set the soc to the next soc
        self.episode.soc = self.episode.next_soc.copy()

        # advance one time step
        self.episode.time += np.timedelta64(self.time_conf.minutes, 'm')

        # get the next observation
        next_obs = self.observer.get_obs(self.db, self.time_conf.price_lookahead, self.episode.time)
        next_obs_soc = next_obs[0]
        next_obs_time_left = next_obs[1]
        next_obs_price = next_obs[2]

        self.episode.price = next_obs_price

        # go through the cars and check whether the same car is still there, no car, or a new car
        for car in range(self.num_cars):

            self.episode.soh -= self.battery_degradation.calculate_cycle_loss(self.episode.old_soc[car], self.episode.soc[car], 11)

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
                self.episode.soh -= self.battery_degradation.calculate_calendar_aging_while_parked(self.episode.old_soc[car], self.episode.soc[car], self.time_conf)

            # no car in the next time step
            elif next_obs_time_left[car] == 0:
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.soc[car] = next_obs_soc[car]

            # new car in the next time step
            elif (self.episode.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.old_soc[car] = self.episode.soc[car]
                self.episode.soc[car] = next_obs_soc[car]
                trip_len = self.observer.get_trip_len(self.db, car, self.episode.time)
                self.episode.soh -= self.battery_degradation.calculate_calendar_aging_on_arrival(trip_len, self.episode.old_soc[car], self.episode.soc[car])

            # this shouldn't happen but if it does, an error is thrown
            else:
                raise TypeError("Observation format not recognized")

        # if the finish time is reached, set done to True
        # The RL agent then resets the environment
        # TODO: do I still experience the last timestep or do I finish when I reach it?
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

        next_obs[0] = self.episode.soc
        next_obs[1] = self.episode.hours_left
        next_obs[2] = self.episode.price

        # here, the reward is already in integer format
        # Todo integer or float?
        return self.normalizer.normalize_obs(next_obs), reward, self.episode.done, False, self.info

    def close(self):
        return 0

    def print(self, action):
        print(f"Timestep: {self.episode.time}")
        print(f"Price: {self.episode.price[0] / 1000} €/kWh")
        print(f"SOC: {self.episode.soc}, Time left: {self.episode.hours_left} hours")
        print(f"Action taken: {action}")
        print(f"Actual charging energy: {self.episode.total_charging_energy} kWh")
        print(f"Charging cost/revenue: {self.episode.current_charging_expense} €")
        print(f"SoH: {self.episode.soh}")
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
