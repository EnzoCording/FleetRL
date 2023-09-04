import os
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Literal
import copy

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
from FleetRL.utils.time_picker.eval_time_picker import EvalTimePicker
from FleetRL.utils.time_picker.time_picker import TimePicker

from FleetRL.utils.battery_degradation.batt_deg import BatteryDegradation
from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
from FleetRL.utils.battery_degradation.rainflow_sei_degradation import RainflowSeiDegradation
from FleetRL.utils.battery_degradation.log_data_deg import LogDataDeg

from FleetRL.utils.data_logger.data_logger import DataLogger

from FleetRL.utils.schedule.schedule_generator import ScheduleGenerator, ScheduleType


class FleetEnv(gym.Env):

    """
    FleetRL: Reinforcement Learning environment for commercial vehicle fleets.
    Author: Enzo Alexander Cording - https://github.com/EnzoCording
    Master's thesis project, M.Sc. Sustainable Energy Engineering @ KTH
    Copyright (c) 2023, Enzo Cording

    This framework is built on the gymnasium core API and inherits from it.
    __init__, reset, and step are implemented, calling other modules and functions where needed.
    Base-derived class architecture is implemented wherever needed, and the code is structured in
    a modular manner to enable improvements, or changes in the model.

    Only publicly available data or own-generated data has been used to implement this model.

    The agent only sees information coming from the chargers: SOC, how long the vehicle is still plugged in, etc.
    However, this framework matches the number of chargers with the number of cars to reduce complexity in modelling.
    If more cars than chargers should be modelled, an allocation algorithm is necessary.
    What is more, battery degradation is modelled in this environment. In this case, the information of the car is
    required (instead of the charger). Modelling is facilitated by matching cars and chargers one-to-one.
    Therefore, throughout the code, car and ev_charger might be used interchangeably as indices.
    """

    def __init__(self,
                 use_case: Literal["ct", "ut", "lmd"],
                 building_name: str,
                 price_name: str,
                 tariff_name: str,
                 schedule_name: str = "lmd_sched_single.csv",
                 pv_name: str = None,
                 include_building: bool = True,
                 include_pv: bool = True,
                 include_price: bool = True,
                 time_picker: Literal["static", "random", "eval"] = "random",
                 target_soc: float = 0.85,
                 max_batt_cap_in_all_use_cases = 60,
                 init_soh: float = 1.0,
                 deg_emp: bool = False,
                 ignore_price_reward=False,
                 ignore_overloading_penalty=False,
                 ignore_invalid_penalty=False,
                 ignore_overcharging_penalty=False,
                 episode_length: int = 48,
                 log_data: bool = True,
                 calculate_degradation: bool = True,
                 verbose: bool = 1,
                 normalize_in_env=False,
                 aux=True,
                 gen_schedule: bool = False,
                 gen_start_date = None,
                 gen_end_date = None,
                 gen_name: str = None,
                 gen_n_evs: int = None,
                 spot_markup: float=None,
                 spot_mul: float=None,
                 feed_in_ded: float=None,
                 seed: int = None
                 ):

        """
        :param schedule_name: String to specify file name of schedule
        :param building_name: String to specify building load data, includes pv as well
        :param pv_name: String to optionally specify own pv dataset
        :param include_building: Flag to include building or not
        :param include_pv: Flag to include pv or not
        :param include_price: Flag to include price or not
        :param time_picker: Specify whether to pick time "static", "random" or "eval" (train vs validation set)
        :param target_soc: Target SOC that needs to be fulfilled before leaving for next trip
        :param max_batt_cap_in_all_use_cases: The largest battery size to be considered in the model
        :param init_soh: Initial state of health of batteries. SOH=1 -> no degradation
        :param deg_emp: Flag to use empirical degradation. Default False
        :param ignore_price_reward: Flag to ignore price reward
        :param ignore_overloading_penalty: Flag to ignore overloading penalty
        :param ignore_invalid_penalty: Flag to ignore invalid action penalty
        :param ignore_overcharging_penalty: Flag to ignore overcharging the battery penalty
        :param episode_length: Length of episode in hours
        :param log_data: Log SOC and SOH to csv files
        :param calculate_degradation: Calculate degradation flag
        :param verbose: Print statements
        :param normalize_in_env: Conduct normalization in environment
        :param use_case: String to specify the use-case
        :param aux: Flag to include auxiliary information in the model
        :param gen_schedule: Flag to generate schedule or not
        :param gen_start_date: Start date of the schedule
        :param gen_end_date: End date of the schedule
        :param gen_name: File name of the schedule
        :param gen_n_evs: How many EVs a schedule should be generated for
        :param spot_markup: markup on the spot price: new_price = spot + X ct/kWh
        :param spot_mul: Multiplied on the price: New price = (spot + markup) * (1+X)
        :param feed_in_ded: Deduction of the feed-in tariff: new_feed_in = (1-X) * feed_in
        :param seed: seed for random number generators
        """

        # call __init__() of parent class to ensure inheritance chain
        super().__init__()

        # setting seed
        self.seed = seed
        np.random.seed(self.seed)

        # Loading configs
        self.time_conf = TimeConfig()
        self.ev_conf = EvConfig()
        self.score_conf = ScoreConfig()

        # Setting flags for the type of environment to build
        # NOTE: observations are appended to the db in the order specified here
        self.include_price = include_price
        self.include_building_load = include_building
        self.include_pv = include_pv
        self.aux_flag = aux  # include auxiliary information

        # conduct normalization of observations
        self.normalize_in_env = normalize_in_env

        # Setting paths and file names
        # path for input files, needs to be the same for all inputs
        self.path_name = os.path.dirname(__file__) + '/../Final_Inputs/'

        # EV schedule database
        # generating own schedules or importing them
        self.generate_schedule = gen_schedule
        self.schedule_name = schedule_name

        # Price databases
        self.spot_name = price_name
        self.tariff_name = tariff_name

        # Building load database
        self.building_name = building_name

        # PV database is the same in this case
        if pv_name is not None:
            self.pv_name = pv_name
        else:
            self.pv_name = building_name

        # Specify company type and associated battery size in kWh
        if use_case == "ct":
            self.company = CompanyType.Caretaker
            self.schedule_type = ScheduleType.Caretaker
            self.ev_conf.init_battery_cap = 16.7
        elif use_case == "ut":
            self.company = CompanyType.Utility
            self.schedule_type = ScheduleType.Utility
            self.ev_conf.init_battery_cap = 50.0
        elif use_case == "lmd":
            self.company = CompanyType.Delivery
            self.schedule_type = ScheduleType.Delivery
            self.ev_conf.init_battery_cap = 60.0
        else:
            raise TypeError("Company not recognised.")

        # adjust this if schedules need to be generated
        if self.generate_schedule:
            gen_sched = []
            if self.seed is not None:
                gen_seed = self.seed
            else:
                gen_seed = 0
            print("Generating schedules... This may take a while.")
            for i in range(gen_n_evs):
                self.schedule_gen = ScheduleGenerator(file_comment=gen_name,
                                                      schedule_dir=self.path_name,
                                                      schedule_type=self.schedule_type,
                                                      starting_date=gen_start_date,
                                                      ending_date=gen_end_date,
                                                      vehicle_id=f"{i}",
                                                      seed=gen_seed + i,
                                                      save_schedule=False)
                gen_sched.append(self.schedule_gen.generate_schedule())

            complete_schedule = pd.concat(gen_sched)
            if not gen_name.endswith(".csv"):
                gen_name = gen_name + ".csv"
            complete_schedule.to_csv(self.path_name + gen_name)
            print(f"Schedule generation complete. Saved in Inputs path. File name: {gen_name}")
            self.schedule_name = gen_name

        # Changing markups if specified
        if spot_markup is not None:
            self.ev_conf.fixed_markup = spot_markup
        if spot_mul is not None:
            self.ev_conf.variable_multiplier = spot_mul
        if feed_in_ded is not None:
            self.ev_conf.feed_in_deduction = feed_in_ded

        # scaling price conf with battery capacity. Each use-case has different battery sizes, so a full charge
        # would have different penalty ranges with different battery capacities. Normalized to max capacity (60 kWh)
        # if different use-cases are compared, change 60 to the highest battery capacity in kWh
        self.score_conf.price_multiplier = (self.score_conf.price_multiplier
                                            * (max_batt_cap_in_all_use_cases / self.ev_conf.init_battery_cap))

        # Changing parameters, if specified
        self.time_conf.episode_length = episode_length
        self.ev_conf.target_soc = target_soc

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
        self.print_updates = verbose
        self.print_reward = verbose
        self.print_function = verbose
        self.calc_deg = calculate_degradation
        self.log_data = log_data

        # Class simulating EV charging
        self.ev_charger: EvCharger = EvCharger(self.ev_conf)

        # Load time picker module
        if time_picker == "static":
            # when an episode starts, this class picks the same starting time
            self.time_picker: TimePicker = StaticTimePicker()
        elif time_picker == "eval":
            # picks a random starting times from test set (nov - dez)
            self.time_picker: TimePicker = EvalTimePicker(self.time_conf.episode_length)
        elif time_picker =="random":
            # picks random starting times from training set (jan - oct)
            self.time_picker: TimePicker = RandomTimePicker()
        else:
            # must choose between static, eval or random
            raise TypeError("Time picker type not recognised")

        # Choose the right observer module based on the environment settings
        # All observations are made in the observer class
        # not even price: only soc and time left
        if not self.include_price:
            self.observer: Observer = ObserverSocTimeOnly()
        # only price
        elif not self.include_building_load and not self.include_pv:
            self.observer: Observer = ObserverPriceOnly()
        # price and building load
        elif self.include_building_load and not self.include_pv:
            self.observer: Observer = ObserverWithBuildingLoad()
        # price and pv
        elif not self.include_building_load and self.include_pv:
            self.observer: Observer = ObserverWithPV()
        # price, building load and pv
        elif self.include_building_load and self.include_pv:
            self.observer: Observer = ObserverWithBoth()

        # Instantiating episode object
        # Episode object contains all episode-specific information
        self.episode: Episode = Episode(self.time_conf)

        # Setting EV parameters
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.initial_soh = init_soh  # initial degree of battery degradation, assumed equal for all cars
        self.min_laxity: float = self.ev_conf.min_laxity  # How much excess time the car should at least have to charge

        # initiating variables inside __init__() that are needed for gym.Env
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)

        # Loading the data logger for battery degradation
        self.deg_data_logger: LogDataDeg = LogDataDeg(self.episode)

        # Loading data logger for analysing results and everything else
        self.data_logger: DataLogger = DataLogger(self.time_conf.episode_length * self.time_conf.time_steps_per_hour)

        # Loading the inputs
        self.data_loader: DataLoader = DataLoader(self.path_name, self.schedule_name,
                                                  self.spot_name, self.tariff_name,
                                                  self.building_name, self.pv_name,
                                                  self.time_conf, self.ev_conf, self.ev_conf.target_soc,
                                                  self.include_building_load, self.include_pv
                                                  )

        # get the total database
        self.db = self.data_loader.db

        # make an adjustment for caretakers: the afternoon tour SOC on arrival should be calculated with the
        # afternoon target SOC. This is set to 0.65 in this case
        if use_case == "ct":
            afternoon_trips = self.db.loc[((self.db["date"].dt.hour >= 0) & (self.db["date"].dt.hour <= 10))
                                          | ((self.db["date"].dt.hour >= 15) & (self.db["date"].dt.hour <= 23))]

            self.db.loc[((self.db["date"].dt.hour >= 0) & (self.db["date"].dt.hour <= 10))
                        | ((self.db["date"].dt.hour >= 15) & (self.db["date"].dt.hour <= 23)), "SOC_on_return"]\
                = (self.ev_conf.target_soc_lunch
                   - afternoon_trips["last_trip_total_consumption"].div(self.ev_conf.init_battery_cap))

            self.db.loc[self.db["There"] == 0, "SOC_on_return"] = 0

        # first ID is 0
        self.num_cars = self.db["ID"].max() + 1

        '''
        # Maximum building load is required to determine grid connection if value is not known.
        # Grid connection is sized at 1.1 times the maximum building load, or such that the charging
        # of 50% of EVs at full capacity causes a grid overloading.
        # This can be changed in the load calculation module, e.g. replacing it with a fixed value.
        '''

        # Target SoC - Vehicles should always leave with this SoC
        self.target_soc = np.ones(self.num_cars) * self.ev_conf.target_soc

        if include_building:
            max_load = max(self.db["load"])
        else:
            max_load = 0  # building load not considered in that case

        # Instantiate load calculation with the necessary information
        self.load_calculation = LoadCalculation(self.company, num_cars=self.num_cars, max_load=max_load)

        # choosing degradation methodology: empirical linear or non-linear mathematical model
        if deg_emp:
            self.emp_deg: BatteryDegradation = EmpiricalDegradation(self.initial_soh, self.num_cars)
        else:
            self.sei_deg: BatteryDegradation = RainflowSeiDegradation(self.initial_soh, self.num_cars)

        # de-trend prices to make them usable as agent rewards
        if self.include_price:
            self.db = DataLoader.shape_price_reward(self.db, self.ev_conf)

        '''
        # Normalizing observations (Oracle) or just concatenating (Unit)
        # Oracle is normalizing with the maximum values, that are assumed to be known
        # Unit doesn't normalize, but just concatenates, and parses data in the right format
        # Auxiliary flag is parsed, to include additional information or not
        # NB: If auxiliary data is changed, the observers, normalizers and dimensions have to be updated
        '''

        if self.normalize_in_env:
            self.normalizer: Normalization = OracleNormalization(self.db,
                                                                 self.include_building_load,
                                                                 self.include_pv,
                                                                 self.include_price,
                                                                 aux=self.aux_flag,
                                                                 ev_conf=self.ev_conf,
                                                                 load_calc=self.load_calculation)
        else:
            self.normalizer: Normalization = UnitNormalization()

        '''
        # set boundaries of the observation space, detects if normalized or not.
        # If aux flag is true, additional information enlarges the observation space.
        # The following code goes through all possible environment setups.
        # Depending on the setup, the dimensions differ and every case is handled differently.
        '''

        if not self.include_price:
            dim = 2 * self.num_cars  # soc and time left
            if self.aux_flag:
                dim += self.num_cars  # there
                dim += self.num_cars  # target soc
                dim += self.num_cars  # charging left
                dim += self.num_cars  # hours needed
                dim += self.num_cars  # laxity
                dim += 1  # evse power
                dim += 6  # month, week, hour sin/cos
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif not self.include_building_load and not self.include_pv:
            dim = 2 * self.num_cars + (self.time_conf.price_lookahead + 1) * 2
            if self.aux_flag:
                dim += self.num_cars  # there
                dim += self.num_cars  # target soc
                dim += self.num_cars  # charging left
                dim += self.num_cars  # hours needed
                dim += self.num_cars  # laxity
                dim += 1  # evse power
                dim += 6  # month, week, hour sin/cos
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif self.include_building_load and not self.include_pv:
            dim = (2 * self.num_cars
                   + (self.time_conf.price_lookahead + 1) * 2
                   + self.time_conf.bl_pv_lookahead + 1
                   )
            if self.aux_flag:
                dim += self.num_cars  # there
                dim += self.num_cars  # target soc
                dim += self.num_cars  # charging left
                dim += self.num_cars  # hours needed
                dim += self.num_cars  # laxity
                dim += 1  # evse power
                dim += 1  # grid cap
                dim += 1  # avail grid cap for charging
                dim += 1  # possible avg action per car
                dim += 6  # month, week, hour sin/co
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif not self.include_building_load and self.include_pv:
            dim = (2 * self.num_cars
                   + (self.time_conf.price_lookahead + 1) * 2
                   + self.time_conf.bl_pv_lookahead + 1
                   )
            if self.aux_flag:
                dim += self.num_cars  # there
                dim += self.num_cars  # target soc
                dim += self.num_cars  # charging left
                dim += self.num_cars  # hours needed
                dim += self.num_cars  # laxity
                dim += 1  # evse power
                dim += 6  # month, week, hour sin/cos
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        elif self.include_building_load and self.include_pv:
            dim = (2 * self.num_cars  # soc and time left
                   + (self.time_conf.price_lookahead + 1) * 2  # price and tariff
                   + 2 * (self.time_conf.bl_pv_lookahead + 1)  # pv and building load
                   )
            if self.aux_flag:
                dim += self.num_cars  # there
                dim += self.num_cars  # target soc
                dim += self.num_cars  # charging left
                dim += self.num_cars  # hours needed
                dim += self.num_cars  # laxity
                dim += 1  # evse power
                dim += 1  # grid cap
                dim += 1  # avail grid cap for charging
                dim += 1  # possible avg action per car
                dim += 6  # month, week, hour sin/cos
            low_obs, high_obs = self.normalizer.make_boundaries(dim)

        else:
            low_obs = None
            high_obs = None
            raise ValueError("Problem with environment setup. Check building and pv flags.")

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cars,), dtype=np.float32)

    def reset(self, **kwargs) -> tuple[np.array, dict]:

        """
        :param kwargs: Necessary for gym inheritance
        :return: First observation (either normalized or not) and an info dict
        """

        # reset degradation logs for new episode
        self.deg_data_logger.log = []
        self.deg_data_logger.soc_log = []

        # set done to False, since the episode just started
        self.episode.done = False

        # instantiate soh - depending on initial health settings
        self.episode.soh = np.ones(self.num_cars) * self.initial_soh

        # based on soh, instantiate battery capacity
        self.episode.battery_cap = self.episode.soh * self.ev_conf.init_battery_cap

        # choose a start time based on the type of choice: same, random, deterministic
        self.episode.start_time = self.time_picker.choose_time(self.db, self.time_conf.freq,
                                                               self.time_conf.end_cutoff
                                                               )

        # calculate the finish time based on the episode length
        self.episode.finish_time = self.episode.start_time + np.timedelta64(self.time_conf.episode_length, 'h')

        # set the model time to the start time
        self.episode.time = self.episode.start_time

        # get observation from observer module
        obs = self.observer.get_obs(self.db,
                                    self.time_conf.price_lookahead,
                                    self.time_conf.bl_pv_lookahead,
                                    self.episode.time,
                                    ev_conf=self.ev_conf,
                                    load_calc=self.load_calculation,
                                    aux=self.aux_flag,
                                    target_soc=self.target_soc)

        # get the first soc and hours_left observation
        self.episode.soc = obs["soc"]
        self.episode.hours_left = obs["hours_left"]
        if self.include_price:
            self.episode.price = obs["price"]
            self.episode.tariff = obs["tariff"]

        ''' if time is insufficient due to unfavourable start date (for example loading an empty car with 15 min
        time left), soc is set in such a way that the agent always has a chance to fulfil the objective'''
        for car in range(self.num_cars):
            p_avail = min([self.ev_conf.obc_max_power, self.load_calculation.evse_max_power])
            time_needed = (self.target_soc[car] - self.episode.soc[car]) * self.episode.battery_cap[car] / p_avail

            # Gives some tolerance, check if hours_left > 0 because car has to be plugged in
            # Makes sure that enough laxity is present, in this case 50% is default
            if (self.episode.hours_left[car] > 0) and (self.ev_conf.min_laxity * time_needed > self.episode.hours_left[car]):
                self.episode.soc[car] = (self.target_soc[car] -
                                         (time_needed * p_avail / self.episode.battery_cap[car]) / self.ev_conf.min_laxity)
                if self.print_updates:
                    print("Initial SOC modified due to unfavourable starting condition.")

        # soc for battery degradation
        self.episode.soc_deg = self.episode.soc.copy()
        # for battery degradation adjust to default soc, if soc is unknown in the beginning
        for car in range(self.num_cars):
            if self.episode.soc_deg[car] == 0:
                self.episode.soc_deg[car] = self.ev_conf.def_soc

        # set the reward history back to an empty list, set cumulative reward to 0
        self.episode.reward_history = []
        self.episode.cumulative_reward = 0
        self.episode.penalty_record = 0

        # rebuild the observation vector with modified values
        obs["soc"] = self.episode.soc
        obs["hours_left"] = self.episode.hours_left
        if self.include_price:
            obs["price"] = self.episode.price
            obs["tariff"] = self.episode.tariff

        # Parse observation to normalization module
        norm_obs = self.normalizer.normalize_obs(obs)

        # Log first soc for battery degradation
        if self.calc_deg:
            self.deg_data_logger.log_soc(self.episode.soc_deg)

        if self.log_data and not self.episode.done:
            # obs action reward cashflow
            self.data_logger.log_data(self.episode.time,
                                      norm_obs,  # normalized observation
                                      np.zeros(self.num_cars),  # action
                                      0.0,  # reward
                                      0.0,  # cashflow
                                      0.0,  # penalties
                                      0.0,  # grid overloading
                                      0.0,  # soc missing on departure
                                      0.0,  # degradation
                                      np.zeros(self.num_cars),  # log of charged energy in kWh
                                      self.episode.soh)  # soh

        return norm_obs, self.info

    def step(self, actions: np.array) -> tuple[np.array, float, bool, bool, dict]:
        """
        :param actions: Actions parsed by the agent
        :return: Tuple containing next observation, reward, done, truncated and info dictionary
        """

        # define variables that are newly used every iteration
        cum_soc_missing = 0  # cumulative soc missing for each step
        there = self.db["There"][self.db["date"] == self.episode.time].values  # plugged in y/n (before next time step)

        # parse the action to the charging function and receive the soc, next soc, reward and cashflow
        self.episode.soc, self.episode.next_soc, reward, cashflow, self.charge_log = self.ev_charger.charge(
            self.db, self.num_cars, actions, self.episode, self.load_calculation,
            self.ev_conf, self.time_conf, self.score_conf, self.print_updates, self.target_soc)

        # set the soc to the next soc
        self.episode.old_soc = self.episode.soc.copy()
        self.episode.soc = self.episode.next_soc.copy()

        # save cashflow for print function
        self.episode.current_charging_expense = cashflow

        # calling the print function
        if self.print_function:
            self.print(actions)

        # check current load and pv for violation check
        if self.include_building_load:
            current_load = self.db.loc[self.db["date"] == self.episode.time, "load"].values[0]
        else:
            current_load = 0

        if self.include_pv:
            current_pv = self.db.loc[self.db["date"] == self.episode.time, "pv"].values[0]
        else:
            current_pv = 0

        # correct actions for spots where no car is plugged in
        corrected_actions = actions * there
        # check if connection has been overloaded and by how much
        overloaded_flag, overload_amount = self.load_calculation.check_violation(corrected_actions,
                                                                                 self.db,
                                                                                 current_load, current_pv)
        relative_loading = overload_amount / self.load_calculation.grid_connection + 1
        # overload_penalty is calculated from a sigmoid function in score_conf
        if overloaded_flag:
            overload_penalty = self.score_conf.overloading_penalty(relative_loading)
            reward += overload_penalty
            self.episode.penalty_record += overload_penalty
            if self.print_updates:
                print(f"Grid connection of {self.load_calculation.grid_connection} kW has been overloaded:"
                      f" {abs(overload_amount)} kW. Penalty: {round(overload_penalty, 3)}")

        # advance one time step
        self.episode.time += np.timedelta64(self.time_conf.minutes, 'm')

        # get the next observation entry from the dataset to get new arrivals or departures
        next_obs = self.observer.get_obs(self.db,
                                         self.time_conf.price_lookahead,
                                         self.time_conf.bl_pv_lookahead,
                                         self.episode.time,
                                         ev_conf=self.ev_conf,
                                         load_calc=self.load_calculation,
                                         aux=self.aux_flag,
                                         target_soc=self.target_soc)
        next_obs_soc = next_obs["soc"]
        next_obs_time_left = next_obs["hours_left"]
        if self.include_price:
            next_obs_price = next_obs["price"]
            self.episode.price = next_obs_price
            next_obs_tariff = next_obs["tariff"]
            self.episode.tariff = next_obs_tariff

        # go through the stations and check whether the same car is still there, no car, or a new arrival
        for car in range(self.num_cars):

            # check if a car just left and didn't fully charge
            if (self.episode.hours_left[car] != 0) and (next_obs_time_left[car] == 0):

                # caretaker is a special case because of the lunch break
                # it is not long enough to fully recharge, so a different target soc is applied
                if self.company == CompanyType.Caretaker:
                    # lunch break case
                    if (self.episode.time.hour > 11) and (self.episode.time.hour < 15):
                        # check for soc violation
                        if self.ev_conf.target_soc_lunch - self.episode.soc[car] > self.eps:
                            # penalty for not fulfilling charging requirement, square difference, scale and clip
                            soc_missing = self.ev_conf.target_soc_lunch - self.episode.soc[car]
                            cum_soc_missing += soc_missing
                            #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                            #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                            current_soc_pen = self.score_conf.soc_violation_penalty(soc_missing)
                            reward += current_soc_pen
                            self.episode.penalty_record += current_soc_pen
                            if self.print_updates:
                                print(f"A car left the station without reaching the target SoC."
                                      f" Penalty: {round(current_soc_pen, 3)}")

                        else: reward += self.score_conf.fully_charged_reward  # reward for fully charging the car

                    # caretaker, other operation times, check for violation
                    elif self.target_soc[car] - self.episode.soc[car] > self.eps:
                        # current_soc_pen is calculated from a sigmoid function in score_conf
                        soc_missing = self.target_soc[car] - self.episode.soc[car]
                        cum_soc_missing += soc_missing
                        #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                        #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                        current_soc_pen = self.score_conf.soc_violation_penalty(soc_missing)
                        reward += current_soc_pen
                        self.episode.penalty_record += current_soc_pen
                        if self.print_updates:
                            print(f"A car left the station without reaching the target SoC."
                                  f" Penalty: {round(current_soc_pen, 3)}")

                    else:
                        reward += self.score_conf.fully_charged_reward  # reward for fully charging the car

                # other companies: if charging requirement wasn't met (with some tolerance eps)
                elif self.target_soc[car] - self.episode.soc[car] > self.eps:
                    # current_soc_pen is calculated from a sigmoid function in score_conf
                    soc_missing = self.target_soc[car] - self.episode.soc[car]
                    cum_soc_missing += soc_missing
                    #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                    #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                    current_soc_pen = self.score_conf.soc_violation_penalty(soc_missing)
                    reward += current_soc_pen
                    self.episode.penalty_record += current_soc_pen
                    if self.print_updates:
                        print(f"A car left the station without reaching the target SoC."
                              f" Penalty: {round(current_soc_pen, 3)}")

                else:
                    reward += self.score_conf.fully_charged_reward  # reward for fully charging the car

            # still charging
            if (next_obs_time_left[car] != 0) and (self.episode.hours_left[car] != 0):
                self.episode.hours_left[car] -= self.time_conf.dt

            # no car in the next time step
            elif next_obs_time_left[car] == 0:
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.soc[car] = next_obs_soc[car]

            # new arrival in the next time step
            elif (self.episode.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                self.episode.hours_left[car] = next_obs_time_left[car]
                self.episode.old_soc[car] = self.episode.soc[car]
                self.episode.soc[car] = next_obs_soc[car]

            # this shouldn't happen but if it does, an error is thrown
            else:
                raise TypeError("Observation format not recognized")

            # if battery degradation >= 10%, target SOC is increased to ensure sufficient kWh in the battery
            if self.episode.soh[car] <= 0.9:
                self.target_soc[car] = 0.9
                if self.print_updates and self.target_soc[car] != 0.9:
                    print(f"Target SOC of Car {car} has been adjusted to 0.9 due to high battery degradation."
                          f"Current SOH: {self.episode.soh[car]}")

        # Update SOH value for degradation calculations, wherever a car is plugged in
        for car in range(self.num_cars):
            if self.episode.hours_left[car] != 0:
                self.episode.soc_deg[car] = self.episode.soc[car]

        # if the finish time is reached, set done to True
        # The RL_agents agent then resets the environment
        if self.episode.time == self.episode.finish_time:
            self.episode.done = True
            if self.calc_deg:
                self.deg_data_logger.add_log_entry()
            if self.print_updates:
                print(f"Episode done: {self.episode.done}")
                self.logged_data = self.data_logger.log

        # append to the reward history
        self.episode.cumulative_reward += reward
        self.episode.reward_history.append((self.episode.time, self.episode.cumulative_reward))

        # TODO: Here could be a saving function that saves the results of the episode

        if self.print_reward:
            print(f"Reward signal: {round(reward, 3)}")
            print("---------")
            print("\n")

        next_obs["soc"] = self.episode.soc
        next_obs["hours_left"] = self.episode.hours_left
        if self.include_price:
            next_obs["price"] = self.episode.price
            next_obs["tariff"] = self.episode.tariff

        # normalize next observation
        norm_next_obs = self.normalizer.normalize_obs(next_obs)

        # Log soc for battery degradation
        if self.calc_deg:
            self.deg_data_logger.log_soc(self.episode.soc_deg)

        # for logging: calculate penalty amount, grid overloading in kW and percentage points of SOC violated
        penalty = reward - (cashflow * self.score_conf.price_multiplier)
        grid = abs(overload_amount)
        soc_v = abs(cum_soc_missing)

        # Calculate degradation and state of health based on chosen method
        # calculate degradation once per day
        if self.calc_deg and ((self.episode.time.hour == 14) and (self.episode.time.minute == 45)):
            degradation = self.sei_deg.calculate_degradation(self.deg_data_logger.soc_log,
                                                             self.load_calculation.evse_max_power,
                                                             self.time_conf,
                                                             self.ev_conf.temperature)
            # calculate SOH from current degradation
            self.episode.soh = np.subtract(self.episode.soh, degradation)
            # calculate new resulting battery capacity after degradation
            self.episode.battery_cap = self.episode.soh * self.ev_conf.init_battery_cap
        # otherwise set degradation to 0 for logging purposes
        else:
            degradation = 0.0

        # log data if episode is not done, otherwise first observation of next episode would be returned
        if self.log_data and not self.episode.done:
            self.data_logger.log_data(self.episode.time,
                                      norm_next_obs,
                                      actions,
                                      reward,
                                      cashflow,
                                      penalty,
                                      grid,
                                      soc_v,
                                      degradation,
                                      self.charge_log,
                                      self.episode.soh)

        # return according to openAI gym core API
        return norm_next_obs, reward, self.episode.done, False, self.info

    def close(self):
        return None

    def print(self, action):
        print(f"Timestep: {self.episode.time}")
        if self.include_price:
            print(f"Total price with fees: {np.round(self.episode.price[0] / 1000, 3)} €/kWh")
            current_spot = self.db.loc[self.db["date"]==self.episode.time, "DELU"].values[0]
            print(f"Spot: {np.round(current_spot/1000, 3)} €/kWh")
            print(f"Tariff: {self.episode.tariff[0] / 1000} €/kWh")
        print(f"SOC: {np.round(self.episode.soc, 3)}, Time left: {self.episode.hours_left} hours")
        print(f"Action taken: {np.round(action, 3)}")
        print(f"Actual charging energy: {round(self.episode.total_charging_energy, 3)} kWh")
        print(f"Logging energy: {round(self.charge_log.sum(), 3)} kWh")
        print(f"Charging cost/revenue: {round(self.episode.current_charging_expense, 3)} €")
        print(f"SoH: {np.round(self.episode.soh, 3)}")
        print("--------------------------")

    def render(self):
        # TODO: graph of rewards for example, or charging power or sth like that
        # TODO: Maybe a bar graph, centered at 0, n bars for n vehicles and height changes with power
        pass

    # functions that can be called through vec_envs via env_method()
    def get_log(self):
        # return log dataframe
        return self.data_logger.log

    def is_done(self):
        # return if episode is done
        return self.episode.done

    def get_start_time(self):
        return self.episode.start_time

    def set_start_time(self, start_time: str):
        self.episode.start_time = start_time
        return None

    def get_time(self):
        return self.episode.time

    def get_dist_factor(self):
        obs = self.observer.get_obs(self.db,
                                    self.time_conf.price_lookahead,
                                    self.time_conf.bl_pv_lookahead,
                                    self.episode.time,
                                    ev_conf=self.ev_conf,
                                    load_calc=self.load_calculation,
                                    aux=self.aux_flag,
                                    target_soc=self.target_soc)

        return np.divide(obs["hours_needed"], np.add(obs["hours_left"], 0.001))
