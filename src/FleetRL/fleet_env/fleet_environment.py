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
from FleetRL.utils.observation.observer_price_only import ObserverPriceOnly
from FleetRL.utils.observation.observer import Observer
from FleetRL.utils.observation.observer_with_pv import ObserverWithPV
from FleetRL.utils.observation.observer_bl_pv import ObserverWithBoth
from FleetRL.utils.observation.observer_soc_time_only import ObserverSocTimeOnly

from FleetRL.utils.time_picker.random_time_picker import RandomTimePicker
from FleetRL.utils.time_picker.static_time_picker import StaticTimePicker
from FleetRL.utils.time_picker.time_picker import TimePicker

from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
from FleetRL.utils.battery_degradation.battery_degradation import BatteryDegradation
from FleetRL.utils.battery_degradation.rainflow_sei_degradation import RainFlowSei

from FleetRL.utils.new_battery_degradation.new_batt_deg import NewBatteryDegradation
from FleetRL.utils.new_battery_degradation.new_empirical_degradation import NewEmpiricalDegradation
from FleetRL.utils.new_battery_degradation.new_rainflow_sei_degradation import NewRainflowSeiDegradation

from FleetRL.utils.data_logger.log_data import DataLogger
from FleetRL.utils.schedule_generator.schedule_generator import ScheduleGenerator, ScheduleType, ScheduleConfig

class FleetEnv(gym.Env):

    def __init__(self, schedule_name:str="lmd_sched_single.csv",
                 building_name:str="load_lmd.csv",
                 include_building:bool=False,
                 include_pv:bool=False,
                 include_price:bool=True,
                 static_time_picker:bool=False,
                 target_soc:float=0.85,
                 init_soh:float=1.0,
                 deg_emp:bool=False,
                 ignore_price_reward = False,
                 ignore_overloading_penalty = False,
                 ignore_invalid_penalty = False,
                 ignore_overcharging_penalty = False,
                 episode_length:int = 24):

        # Setting paths and file names
        # path for input files, needs to be the same for all inputs
        self.path_name = os.path.dirname(__file__) + '/../Final_Inputs/'

        # EV schedule database
        # generating own schedules or importing them
        self.generate_schedule = False
        # self.schedule_name = "schedule_1685458125_one_year_15_min_delivery.csv"
        self.schedule_name = schedule_name
        # self.schedule_name = 'full_test_one_car.csv'
        # self.schedule_name = 'one_day_same_training.csv'

        # Spot price database
        self.spot_name = 'spot_2020.csv'

        # Building load database
        self.building_name = building_name

        # PV database is the same in this case
        self.pv_name = building_name

        if self.generate_schedule:

            self.schedule_gen = ScheduleGenerator(file_comment="one_year_15_min_delivery",
                                                  schedule_dir=self.path_name,
                                                  schedule_type=ScheduleType.Delivery,
                                                  ending_date="30/12/2023")

            self.schedule_gen.generate_schedule()
            # self.schedule_gen.generate_multiple_ev_schedule(num_evs=1)

            self.schedule_name = self.schedule_gen.get_file_name()


        # Setting flags for the type of environment to build
        # NOTE: they are appended to the db in the order specified here
        # NOTE: import the right observer!
        self.include_building_load = include_building
        self.include_pv = include_pv
        self.include_price = include_price

        # Loading configs
        self.time_conf = TimeConfig()
        self.ev_conf = EvConfig()
        self.score_conf = ScoreConfig()
        self.load_calculation = LoadCalculation(CompanyType.Caretaker)

        # Changing TimeConfig, if specified
        if episode_length > 24:
            self.time_conf.episode_length = episode_length

        # Changing ScoreConfig, if specified
        if ignore_price_reward:
            self.score_conf.price_multiplier = 0
        if ignore_overloading_penalty:
            self.score_conf.penalty_overloading = 0
        if ignore_invalid_penalty:
            self.score_conf.penalty_invalid_action = 0
        if ignore_overcharging_penalty:
            self.score_conf.penalty_overcharging = 0

        # Set printing and logging parameters, false can increase training fps
        self.print_updates = True
        self.print_reward = True
        self.print_function = True
        self.logging = True
        self.log_to_csv = False

        # Loading modules
        self.ev_charger: EvCharger = EvCharger()  # class simulating EV charging
        if static_time_picker:
            self.time_picker: TimePicker = StaticTimePicker()  # when an episode starts, this class picks the same starting time
        else:
            self.time_picker: TimePicker = RandomTimePicker()  # picks random starting times

        # not even price: only soc and time left
        if not self.include_price:
            self.observer: Observer = ObserverSocTimeOnly()
        # only price
        elif not self.include_building_load and not self.include_pv:
            self.observer: Observer = ObserverPriceOnly()
        # price and building load
        elif self.include_building_load and not self.include_pv:
            self.observer: Observer = ObserverWithBuildingLoad()  # all observations are processed in the Observer class
        # price and pv
        elif not self.include_building_load and self.include_pv:
            self.observer: Observer = ObserverWithPV()
        # price, building load and pv
        elif self.include_building_load and self.include_pv:
            self.observer: Observer = ObserverWithBoth()
        self.episode: Episode = Episode(self.time_conf)  # Episode object contains all episode-specific information

        # Setting EV parameters
        self.target_soc = target_soc  # Target SoC - Vehicles should always leave with this SoC
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.initial_soh = init_soh  # initial degree of battery degradation, assumed equal for all cars

        # initiating variables inside __init__() that are needed for gym.Env
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)
        self.max_time_left: float = None  # The maximum value of time left in the db dataframe
        self.max_spot: float = None  # The maximum spot market price in the db dataframe

        # Loading the data logger
        self.data_logger: DataLogger = DataLogger(self.episode)

        # Loading the inputs
        self.data_loader: DataLoader = DataLoader(self.path_name, self.schedule_name,
                                                  self.spot_name, self.building_name, self.pv_name,
                                                  self.time_conf, self.ev_conf, self.target_soc,
                                                  self.include_building_load, self.include_pv
                                                  )

        # get the total database
        self.db = self.data_loader.db

        # first ID is 0
        self.num_cars = self.db["ID"].max() + 1

        self.episode.soh = np.ones(self.num_cars) * self.initial_soh  # initialize soh for each battery

        if deg_emp:
            self.new_emp_batt: NewBatteryDegradation = NewEmpiricalDegradation(self.initial_soh, self.num_cars)
        else:
            self.new_battery_degradation: NewBatteryDegradation = NewRainflowSeiDegradation(self.initial_soh, self.num_cars)

        # Load gym observation spaces, decided which normalization strategy to choose
        self.normalizer: Normalization = OracleNormalization(self.db, self.include_building_load, self.include_pv, self.include_price)
        # unit normalization: doesn't normalize, only concatenates
        # self.normalizer: Normalization = \
        #     UnitNormalization(self.db, self.spot_price, self.num_cars,
        #                       self.time_conf.price_lookahead, self.time_conf.time_steps_per_hour)

        # set boundaries of the observation space, detects if normalized or not
        if not self.include_price:
            dim = 2 * self.num_cars
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif not self.include_building_load and not self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif self.include_building_load and not self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour + self.time_conf.bl_pv_lookahead
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif not self.include_building_load and self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour + self.time_conf.bl_pv_lookahead
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif self.include_building_load and self.include_pv:
            dim = 2 * self.num_cars + self.time_conf.price_lookahead * self.time_conf.time_steps_per_hour + 2 * self.time_conf.bl_pv_lookahead
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        else:
            low_obs = None
            high_obs = None
            raise ValueError("Problem with components. Check building and pv flags.")

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cars,), dtype=np.float32)

    def reset(self, **kwargs) -> tuple[np.array, dict]:

        # reset logs for new episode
        self.data_logger.log = []
        self.data_logger.soc_log = []
        self.data_logger.soh_log = []

        # set done to False, since the episode just started
        self.episode.done = False

        # instantiate soh - depending on initial health settings
        self.episode.soh = np.ones(self.num_cars) * self.initial_soh

        # choose a start time based on the type of choice: same, random, deterministic
        self.episode.start_time = self.time_picker.choose_time(self.db, self.time_conf.freq,
                                                               self.time_conf.end_cutoff
                                                               )

        # calculate the finish time based on the episode length
        self.episode.finish_time = self.episode.start_time + np.timedelta64(self.time_conf.episode_length, 'h')

        # set the model time to the start time
        self.episode.time = self.episode.start_time

        # get observation from observer module
        obs = self.observer.get_obs(self.db, self.time_conf.price_lookahead, self.time_conf.bl_pv_lookahead, self.episode.time)

        # get the first soc and hours_left observation
        self.episode.soc = obs[0] * self.episode.soh
        self.episode.hours_left = obs[1]
        if self.include_price:
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
                if self.print_updates:
                    print("Initial SOC modified due to unfavourable starting condition.")

            if self.episode.soc[car] == 0:
                self.episode.soc[car] = self.ev_conf.def_soc
                if self.print_updates:
                    print(f"Initial SoC changed from 0 to default value: {self.ev_conf.def_soc}")

        # set the reward history back to an empty list, set cumulative reward to 0
        self.episode.reward_history = []
        self.episode.cumulative_reward = 0
        self.episode.penalty_record = 0

        obs[0] = self.episode.soc
        obs[1] = self.episode.hours_left
        if self.include_price:
            obs[2] = self.episode.price

        if self.logging:
            self.data_logger.log_soc(self.episode)
            self.data_logger.log_soh(self.episode)

        # TODO Unit normalizer boundaries
        return self.normalizer.normalize_obs(obs), self.info

    def step(self, actions: np.array) -> tuple[np.array, float, bool, bool, dict]:

        # parse the action to the charging function and receive the soc, next soc, reward and cashflow
        # the soh is taken into account within the charge function
        self.episode.soc, self.episode.next_soc, reward, cashflow = self.ev_charger.charge(
            self.db, self.num_cars, actions, self.episode, self.load_calculation,
            self.ev_conf, self.time_conf, self.score_conf, self.print_updates, self.target_soc)

        # save the old soc for logging purposes
        self.episode.old_soc = self.episode.soc

        # at this point, the reward only includes the current expense/revenue of the charging process
        # not true anymore, violation of invalid charging
        # TODO: decouple reward and economic metrics
        # todo remove this
        self.episode.current_charging_expense = reward

        # calling the print function
        if self.print_function:
            self.print(actions)

        # ToDo
        # print(f"Max possible: {self.grid_connection} kW" ,f"Actual: {sum(actions) * self.evse_max_power + self.building_load} kW")
        # print(self.price[0])
        # print(self.hours_left)

        # check if the current load exceeds the trafo rating and penalize accordingly

        if self.include_building_load:
            current_load = self.db.loc[self.db["date"] == self.episode.time, "load"].values[0]
        else:
            current_load = 0

        if self.include_pv:
            current_pv = self.db.loc[self.db["date"] == self.episode.time, "pv"].values[0]
        else:
            current_pv = 0

        if not self.load_calculation.check_violation(current_load, actions, current_pv):
            reward += self.score_conf.penalty_overloading
            self.episode.penalty_record += self.score_conf.penalty_overloading
            if self.print_updates:
                print(f"Grid connection has been overloaded.")

        # set the soc to the next soc
        self.episode.soc = self.episode.next_soc.copy()

        # advance one time step
        self.episode.time += np.timedelta64(self.time_conf.minutes, 'm')

        # get the next observation
        next_obs = self.observer.get_obs(self.db, self.time_conf.price_lookahead, self.time_conf.bl_pv_lookahead, self.episode.time)
        next_obs_soc = next_obs[0]
        next_obs_time_left = next_obs[1]

        if self.include_price:
            next_obs_price = next_obs[2]

            self.episode.price = next_obs_price

        # go through the cars and check whether the same car is still there, no car, or a new car
        for car in range(self.num_cars):

            #self.episode.soh -= self.battery_degradation.calculate_cycle_loss(self.episode.old_soc[car], self.episode.soc[car], 11)

            # check if a car just left and didn't fully charge
            if (self.episode.hours_left[car] != 0) and (next_obs_time_left[car] == 0):
                if self.target_soc - self.episode.soc[car] > self.eps:
                    # TODO could scale with difference to SoC
                    # penalty for not fulfilling charging requirement
                    # TODO: this could be changed. No target SoC, but penalty if a car runs empty on a trip
                    current_soc_pen = self.score_conf.penalty_soc_violation * (self.target_soc - self.episode.soc[car]) ** 2
                    reward += current_soc_pen
                    self.episode.penalty_record += current_soc_pen
                    if self.print_updates:
                        print(f"A car left the station without reaching the target SoC. Penalty: {current_soc_pen}")

            # same car in the next time step
            if (next_obs_time_left[car] != 0) and (self.episode.hours_left[car] != 0):
                self.episode.hours_left[car] -= self.time_conf.dt
                #self.episode.soh -= self.battery_degradation.calculate_calendar_aging_while_parked(self.episode.old_soc[car], self.episode.soc[car], self.time_conf)

            # no car in the next time step
            elif next_obs_time_left[car] == 0:
                self.episode.hours_left[car] = next_obs_time_left[car]
                # self.episode.soc[car] = next_obs_soc[car]
                # instead of 0, leave soc at last known soc
                self.episode.soc[car] = self.episode.old_soc[car]

            # new car in the next time step
            elif (self.episode.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.old_soc[car] = self.episode.soc[car]
                self.episode.soc[car] = next_obs_soc[car]
                trip_len = self.observer.get_trip_len(self.db, car, self.episode.time)
                #self.episode.soh -= self.battery_degradation.calculate_calendar_aging_on_arrival(trip_len, self.episode.old_soc[car], self.episode.soc[car])

            # this shouldn't happen but if it does, an error is thrown
            else:
                raise TypeError("Observation format not recognized")

        # if the finish time is reached, set done to True
        # The RL agent then resets the environment
        # TODO: do I still experience the last timestep or do I finish when I reach it?
        if self.episode.time == self.episode.finish_time:
            self.episode.done = True
            if self.logging:
                self.data_logger.add_log_entry()
            if self.print_updates:
                print(f"Episode done: {self.episode.done}")
            if self.log_to_csv:
                self.data_logger.permanent_log()

        # append to the reward history
        self.episode.cumulative_reward += reward
        self.episode.reward_history.append((self.episode.time, self.episode.cumulative_reward))

        # TODO: Here could be a saving function that saves the results of the episode

        if self.print_reward:
            print(f"Reward signal: {reward}")
            print("---------")
            print("\n")

        next_obs[0] = self.episode.soc
        next_obs[1] = self.episode.hours_left
        if self.include_price:
            next_obs[2] = self.episode.price

        if self.logging:
            self.data_logger.log_soc(self.episode)

        self.episode.soh -= self.new_battery_degradation.calculate_degradation(self.data_logger.soc_log, self.load_calculation.evse_max_power, self.time_conf, self.ev_conf.temperature)

        if self.logging:
            self.data_logger.log_soh(self.episode)

        # here, the reward is already in integer format
        # Todo integer or float?
        return self.normalizer.normalize_obs(next_obs), reward, self.episode.done, False, self.info

    def close(self):
        return 0

    def print(self, action):
        print(f"Timestep: {self.episode.time}")
        if self.include_price:
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
