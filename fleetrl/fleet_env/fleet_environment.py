import os
import json
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Literal
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fleetrl.fleet_env.config.ev_config import EvConfig
from fleetrl.fleet_env.config.score_config import ScoreConfig
from fleetrl.fleet_env.config.time_config import TimeConfig

from fleetrl.fleet_env.episode import Episode

from fleetrl.utils.data_processing.data_processing import DataLoader
from fleetrl.utils.ev_charging.ev_charger import EvCharger
from fleetrl.utils.load_calculation.load_calculation import LoadCalculation, CompanyType

from fleetrl.utils.normalization.normalization import Normalization
from fleetrl.utils.normalization.oracle_normalization import OracleNormalization
from fleetrl.utils.normalization.unit_normalization import UnitNormalization

from fleetrl.utils.observation.observer_with_building_load import ObserverWithBuildingLoad
from fleetrl.utils.observation.observer_price_only import ObserverPriceOnly
from fleetrl.utils.observation.observer import Observer
from fleetrl.utils.observation.observer_with_pv import ObserverWithPV
from fleetrl.utils.observation.observer_bl_pv import ObserverWithBoth
from fleetrl.utils.observation.observer_soc_time_only import ObserverSocTimeOnly

from fleetrl.utils.time_picker.random_time_picker import RandomTimePicker
from fleetrl.utils.time_picker.static_time_picker import StaticTimePicker
from fleetrl.utils.time_picker.eval_time_picker import EvalTimePicker
from fleetrl.utils.time_picker.time_picker import TimePicker

from fleetrl.utils.battery_degradation.batt_deg import BatteryDegradation
from fleetrl.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
from fleetrl.utils.battery_degradation.rainflow_sei_degradation import RainflowSeiDegradation
from fleetrl.utils.battery_degradation.log_data_deg import LogDataDeg

from fleetrl.utils.event_manager.event_manager import EventManager

from fleetrl.utils.data_logger.data_logger import DataLogger

from fleetrl.utils.schedule.schedule_generator import ScheduleGenerator, ScheduleType

from fleetrl.utils.rendering.render import ParkingLotRenderer

class FleetEnv(gym.Env):

    """
    FleetRL: Reinforcement Learning environment for commercial vehicle fleets.
    Author: Enzo Alexander Cording - https://github.com/EnzoCording
    Master's thesis project, M.Sc. Sustainable Energy Engineering @ KTH
    Copyright (c) 2023, Enzo Cording

    This framework is built on the gymnasium core API and inherits from it.
    __init__, reset, and step are implemented, calling other modules and functions where needed.
    Base-derived class architecture is implemented, and the code is structured in
    a modular manner to enable improvements or changes in the model.

    Only publicly available data or own-generated data has been used in this implementation.

    The agent only sees information coming from the chargers: SOC, how long the vehicle is still plugged in, etc.
    However, this framework matches the number of chargers with the number of cars to reduce complexity.
    If more cars than chargers should be modelled, an allocation algorithm is necessary.
    What is more, battery degradation is modelled in this environment. In this case, the information of the car is
    required (instead of the charger). Modelling is facilitated by matching cars and chargers one-to-one.
    Therefore, throughout the code, "car" and "ev_charger" might be used interchangeably as indices.

    Note that this does not present a simplification from the agent perspective because the agent does only handles
    the SOC and time left at the charger, regardless of whether the vehicle is matching the charger one-to-one or not.
    """

    def __init__(self, env_config: str | dict):

        """
        :param env_config: String to specify path of json config file, or dict with config

        The following items are to be specified in the json or dict config:
        - data_path: String to specify the absolute path of the input folder
        - schedule_name: String to specify file name of schedule
        - building_name: String to specify building load data, includes pv as well
        - pv_name: String to optionally specify own pv dataset
        - include_building: Flag to include building or not
        - include_pv: Flag to include pv or not
        - include_price: Flag to include price or not
        - time_picker: Specify whether to pick time "static", "random" or "eval"
        - target_soc: Target SOC that needs to be fulfilled before leaving for next trip
        - max_batt_cap_in_all_use_cases: The largest battery size to be considered in the model
        - init_soh: Initial state of health of batteries. SOH=1 -> no degradation
        - deg_emp: Flag to use empirical degradation. Default False
        - ignore_price_reward: Flag to ignore price reward
        - ignore_overloading_penalty: Flag to ignore overloading penalty
        - ignore_invalid_penalty: Flag to ignore invalid action penalty
        - ignore_overcharging_penalty: Flag to ignore overcharging the battery penalty
        - episode_length: Length of episode in hours
        - log_data: Log SOC and SOH to csv files
        - calculate_degradation: Calculate degradation flag
        - verbose: Print statements
        - normalize_in_env: Conduct normalization in environment
        - use_case: String to specify the use-case
        - aux: Flag to include auxiliary information in the model
        - gen_schedule: Flag to generate schedule or not
        - gen_start_date: Start date of the schedule
        - gen_end_date: End date of the schedule
        - gen_name: File name of the schedule
        - gen_n_evs: How many EVs a schedule should be generated for
        - spot_markup: markup on the spot price: new_price = spot + X ct/kWh
        - spot_mul: Multiplied on the price: New price = (spot + markup) * (1+X)
        - feed_in_ded: Deduction of the feed-in tariff: new_feed_in = (1-X) * feed_in
        - seed: seed for random number generators
        - real_time Bool for specifying real time flag
        """

        # call __init__() of parent class to ensure inheritance chain
        super().__init__()

        # Check that the input parameter config is passed properly - either as json or dict
        assert (env_config.__class__ == dict) or (env_config.__class__ == str), 'Invalid config type.'
        if env_config.__class__ == str:
            assert os.path.isfile(env_config), f'Config file not found at {env_config}.'
            self.env_config = self.read_config(conf_path=env_config)
        else:
            self.env_config = env_config

        # setting seed
        self.seed = self.env_config["seed"]
        np.random.seed(self.seed)

        # Loading configs
        self.time_conf = TimeConfig(self.env_config)
        self.ev_config = EvConfig(self.env_config)
        self.score_config = ScoreConfig(self.env_config)

        # Setting flags for the type of environment to build
        # NOTE: observations are appended to the db in the order specified here
        self.include_price = self.env_config["include_price"]
        self.include_building_load = self.env_config["include_building"]
        self.include_pv = self.env_config["include_pv"]
        self.aux_flag = self.env_config["aux"]  # include auxiliary information

        # conduct normalization of observations
        self.normalize_in_env = self.env_config["normalize_in_env"]

        # Setting paths and file names
        # path for input files, needs to be the same for all inputs
        self.path_name = self.env_config["data_path"]

        # EV schedule database
        # generating own schedules or importing them
        self.generate_schedule = self.env_config["gen_schedule"]
        self.schedule_name = self.env_config["schedule_name"]
        self.gen_name = self.env_config["gen_name"]
        self.gen_start_date = self.env_config["gen_start_date"]
        self.gen_end_date = self.env_config["gen_end_date"]
        self.gen_n_evs = self.env_config["gen_n_evs"]

        # Price databases
        self.spot_name = self.env_config["price_name"]
        self.tariff_name = self.env_config["tariff_name"]

        # Building load database
        self.building_name = self.env_config["building_name"]

        # PV database is the same in this case
        if self.env_config["pv_name"] is not None:
            self.pv_name = self.env_config["pv_name"]
        else:
            self.pv_name = self.env_config["building_name"]

        use_case = self.env_config["use_case"]

        # Specify the company type and size of the battery
        self.company: CompanyType = None
        self.schedule_type: ScheduleType = None
        self.specify_company_and_battery_size(use_case)

        # Automatic schedule generation if specified
        if self.generate_schedule:
            self.auto_gen()

        # Make sure that data paths are correct and point to existing files
        self.check_data_paths(self.path_name, self.schedule_name, self.spot_name, self.building_name, self.pv_name)

        # Changing markups on spot prices if specified in config file (e.g. 20% on top on spot prices)
        self.change_markups()

        # scaling price conf with battery capacity. Each use-case has different battery sizes, so a full charge
        # would have different penalty ranges with different battery capacities. Normalized to max capacity (60 kWh)
        # if different use-cases are compared, change max_batt_cap to the highest battery capacity in kWh
        self.max_batt_cap_in_all_use_cases = self.env_config["max_batt_cap_in_all_use_cases"]
        self.score_config.price_multiplier = (self.score_config.price_multiplier
                                              * (self.max_batt_cap_in_all_use_cases / self.ev_config.init_battery_cap))

        # Changing parameters, if specified
        self.time_conf.episode_length = self.env_config["episode_length"]
        self.ev_config.target_soc = self.env_config["target_soc"]

        # Changing ScoreConfig, if specified, e.g. setting some penalties to zero
        self.adjust_score_config()

        verbose = self.env_config["verbose"]
        # Set printing and logging parameters, false can increase training fps
        self.print_updates = verbose
        self.print_reward = verbose
        self.print_function = verbose

        self.calc_deg = self.env_config["calculate_degradation"]
        self.log_data = self.env_config["log_data"]

        # Event manager to check if a relevant event took place to pass to the agent
        self.event_manager: EventManager = EventManager()

        # Class simulating EV charging
        self.ev_charger: EvCharger = EvCharger(self.ev_config)

        # Choose time picker based on input string time_picker
        self.time_picker = self.choose_time_picker(self.env_config["time_picker"])

        # Choose the right observer module based on the environment settings
        self.observer = self.choose_observer()

        # Instantiating episode object
        # Episode object contains all episode-specific information
        self.episode: Episode = Episode(self.time_conf)

        # Setting EV parameters
        self.eps = 0.005  # allowed SOC deviation from target: 0.5%
        self.initial_soh = self.env_config["init_soh"]  # initial degree of battery degradation, assumed equal for all cars
        self.min_laxity: float = self.ev_config.min_laxity  # How much excess time the car should at least have to charge

        # initiating variables inside __init__() that are needed for gym.Env
        self.info: dict = {}  # Necessary for gym env (Double check because new implementation doesn't need it)

        # Loading the data logger for battery degradation
        self.deg_data_logger: LogDataDeg = LogDataDeg(self.episode)

        # Loading data logger for analysing results and everything else
        self.data_logger: DataLogger = DataLogger(self.time_conf.episode_length * self.time_conf.time_steps_per_hour)

        self.real_time = self.env_config["real_time"]

        # Loading the inputs
        self.data_loader: DataLoader = DataLoader(self.path_name, self.schedule_name,
                                                  self.spot_name, self.tariff_name,
                                                  self.building_name, self.pv_name,
                                                  self.time_conf, self.ev_config, self.ev_config.target_soc,
                                                  self.include_building_load, self.include_pv, self.real_time
                                                  )

        # get the total database
        self.db = self.data_loader.db

        if use_case == "ct":
            self.adjust_caretaker_lunch_soc()

        # first ID is 0
        self.num_cars = self.db["ID"].max() + 1

        # Target SoC - Vehicles should always leave with this SoC
        self.target_soc: np.ndarray = np.ones(self.num_cars) * self.ev_config.target_soc

        if self.env_config["include_building"]:
            max_load = max(self.db["load"])
        else:
            max_load = 0  # building load not considered in that case

        # Instantiate load calculation with the necessary information
        """
        Note:
        - Maximum building load is required to determine grid connection if value is not known.
        - Grid connection is sized at 1.1 times the maximum building load, or such that the charging
        - of 50% of EVs at full capacity causes a grid overloading.
        - This can be changed in the load calculation module, e.g. replacing it with a fixed value.
        """

        self.load_calculation = LoadCalculation(env_config=self.env_config,
                                                company_type=self.company,
                                                num_cars=self.num_cars,
                                                max_load=max_load)

        # choosing degradation methodology: empirical linear or non-linear mathematical model
        if self.env_config["deg_emp"]:
            self.emp_deg: BatteryDegradation = EmpiricalDegradation(self.initial_soh, self.num_cars)
        else:
            self.sei_deg: BatteryDegradation = RainflowSeiDegradation(self.initial_soh, self.num_cars)

        # de-trend prices to make them usable as agent rewards
        if self.include_price:
            self.db = DataLoader.shape_price_reward(self.db, self.ev_config)

        """
        - Normalizing observations (Oracle) or just concatenating (Unit)
        - Oracle is normalizing with the maximum values, that are assumed to be known
        - Unit doesn't normalize, but just concatenates, and parses data in the right format
        - Auxiliary flag is parsed, to include additional information or not
        - NB: If auxiliary data is changed, the observers, normalizers and dimensions have to be updated
        """

        if self.normalize_in_env:
            self.normalizer: Normalization = OracleNormalization(self.db,
                                                                 self.include_building_load,
                                                                 self.include_pv,
                                                                 self.include_price,
                                                                 aux=self.aux_flag,
                                                                 ev_conf=self.ev_config,
                                                                 load_calc=self.load_calculation)
        else:
            self.normalizer: Normalization = UnitNormalization()

        # choose dimensions and bounds depending on settings
        low_obs, high_obs = self.detect_dim_and_bounds()

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32)

        # the action space is also continuous: -1 and 1 being the bounds (-100% to 100% of the EVSE kW power rating)
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cars,), dtype=np.float32)

        self.render_mode = "human"
        self.pl_render: ParkingLotRenderer = ParkingLotRenderer()

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
        self.episode.battery_cap = self.episode.soh * self.ev_config.init_battery_cap

        # choose a start time based on the type of choice: same, random, deterministic
        self.episode.start_time = self.time_picker.choose_time(self.db, self.time_conf.freq,
                                                               self.time_conf.end_cutoff)

        # calculate the finish time based on the episode length
        self.episode.finish_time = self.episode.start_time + np.timedelta64(self.time_conf.episode_length, 'h')

        # set the model time to the start time
        self.episode.time = self.episode.start_time

        # get observation from observer module
        obs = self.observer.get_obs(self.db,
                                    self.time_conf.price_lookahead,
                                    self.time_conf.bl_pv_lookahead,
                                    self.episode.time,
                                    ev_conf=self.ev_config,
                                    load_calc=self.load_calculation,
                                    aux=self.aux_flag,
                                    target_soc=self.target_soc)

        # get the first soc and hours_left observation
        self.episode.soc = obs["soc"]
        self.episode.hours_left = obs["hours_left"]
        if self.include_price:
            self.episode.price = obs["price"]
            self.episode.tariff = obs["tariff"]

        """
        if time is insufficient due to unfavourable start date (for example loading an empty car with 15 min
        time left), soc is set in such a way that the agent always has a chance to fulfil the objective
        """

        for car in range(self.num_cars):
            p_avail = min([self.ev_config.obc_max_power, self.load_calculation.evse_max_power])
            time_needed = (self.target_soc[car] - self.episode.soc[car]) * self.episode.battery_cap[car] / p_avail

            # Gives some tolerance, check if hours_left > 0 because car has to be plugged in
            # Makes sure that enough laxity is present, in this case 50% is default
            if (self.episode.hours_left[car] > 0) and (self.ev_config.min_laxity * time_needed > self.episode.hours_left[car]):
                self.episode.soc[car] = (self.target_soc[car] -
                                         (time_needed * p_avail / self.episode.battery_cap[car]) / self.ev_config.min_laxity)
                if self.print_updates:
                    print("Initial SOC modified due to unfavourable starting condition.")

        # soc for battery degradation
        self.episode.soc_deg = self.episode.soc.copy()
        # for battery degradation adjust to default soc, if soc is unknown in the beginning
        for car in range(self.num_cars):
            if self.episode.soc_deg[car] == 0:
                self.episode.soc_deg[car] = self.ev_config.def_soc

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
        The main logic of the EV charging problem is orchestrated in the step function.
        Input: Action on charging power for each EV
        Output: Next state, reward

        Intermediate processes: EV charging model, battery degradation, cost calculation, building load, penalties, etc.

        The step function runs as long as the done flag is False. Different functions and modules are called in this
        function to reduce the complexity and to distribute the tasks of the model.

        :param actions: Actions parsed by the agent, from -1 to 1, representing % of kW of the EVSE
        :return: Tuple containing next observation, reward, done, truncated and info dictionary
        """

        self.episode.current_actions = actions

        while True:

            self.episode.time_conf.dt = self.get_next_dt()  # get next dt in case time frequency changes
            self.episode.time_conf.time_steps_per_hour = int(1 / np.copy(self.episode.time_conf.dt))
            self.episode.time_conf.minutes = self.get_next_minutes()  # get next minutes in case time freq changes

            # define variables that are newly used every iteration
            cum_soc_missing = 0  # cumulative soc missing for each step
            there = self.db["There"][self.db["date"] == self.episode.time].values  # plugged in y/n (before next time step)

            # parse the action to the charging function and receive the soc, next soc, reward and cashflow
            self.episode.soc, self.episode.next_soc, reward, cashflow, self.charge_log, self.episode.events = self.ev_charger.charge(
                self.db, self.num_cars, actions, self.episode, self.load_calculation,
                self.ev_config, self.time_conf, self.score_config, self.print_updates, self.target_soc)

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
                self.episode.events += 1  # relevant event detected
                overload_penalty = self.score_config.overloading_penalty(relative_loading)
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
                                             ev_conf=self.ev_config,
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

                # checks if a car just left and if rules were violated, e.g. didn't fully charge
                if (self.episode.hours_left[car] != 0) and (next_obs_time_left[car] == 0):
                    self.episode.events += 1  # relevant event detected

                    # caretaker is a special case because of the lunch break
                    # it is not long enough to fully recharge, so a different target soc is applied
                    if self.company == CompanyType.Caretaker:
                        # lunch break case
                        if (self.episode.time.hour > 11) and (self.episode.time.hour < 15):
                            # check for soc violation
                            if self.ev_config.target_soc_lunch - self.episode.soc[car] > self.eps:
                                # penalty for not fulfilling charging requirement, square difference, scale and clip
                                self.episode.events += 1  # relevant event detected
                                soc_missing = self.ev_config.target_soc_lunch - self.episode.soc[car]
                                cum_soc_missing += soc_missing
                                #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                                #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                                current_soc_pen = self.score_config.soc_violation_penalty(soc_missing)
                                reward += current_soc_pen
                                self.episode.penalty_record += current_soc_pen
                                if self.print_updates:
                                    print(f"A car left the station without reaching the target SoC."
                                          f" Penalty: {round(current_soc_pen, 3)}")

                            else: reward += self.score_config.fully_charged_reward  # reward for fully charging the car

                        # caretaker, other operation times, check for violation
                        elif self.target_soc[car] - self.episode.soc[car] > self.eps:
                            # current_soc_pen is calculated from a sigmoid function in score_conf
                            self.episode.events += 1  # relevant event detected
                            soc_missing = self.target_soc[car] - self.episode.soc[car]
                            cum_soc_missing += soc_missing
                            #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                            #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                            current_soc_pen = self.score_config.soc_violation_penalty(soc_missing)
                            reward += current_soc_pen
                            self.episode.penalty_record += current_soc_pen
                            if self.print_updates:
                                print(f"A car left the station without reaching the target SoC."
                                      f" Penalty: {round(current_soc_pen, 3)}")

                        else:
                            reward += self.score_config.fully_charged_reward  # reward for fully charging the car

                    # other companies: if charging requirement wasn't met (with some tolerance eps)
                    elif self.target_soc[car] - self.episode.soc[car] > self.eps:
                        self.episode.events += 1  # relevant event detected
                        # current_soc_pen is calculated from a sigmoid function in score_conf
                        soc_missing = self.target_soc[car] - self.episode.soc[car]
                        cum_soc_missing += soc_missing
                        #current_soc_pen = self.score_conf.penalty_soc_violation * soc_missing ** 2
                        #current_soc_pen = max(current_soc_pen, self.score_conf.clip_soc_violation)
                        current_soc_pen = self.score_config.soc_violation_penalty(soc_missing)
                        reward += current_soc_pen
                        self.episode.penalty_record += current_soc_pen
                        if self.print_updates:
                            print(f"A car left the station without reaching the target SoC."
                                  f" Penalty: {round(current_soc_pen, 3)}")

                    else:
                        reward += self.score_config.fully_charged_reward  # reward for fully charging the car

                # still charging
                if (next_obs_time_left[car] != 0) and (self.episode.hours_left[car] != 0):
                    self.episode.hours_left[car] -= self.time_conf.dt

                # no car in the next time step
                elif next_obs_time_left[car] == 0:
                    self.episode.hours_left[car] = next_obs_time_left[car]
                    self.episode.soc[car] = next_obs_soc[car]

                # new arrival in the next time step
                elif (self.episode.hours_left[car] == 0) and (next_obs_time_left[car] != 0):
                    self.episode.events += 1  # relevant event
                    self.episode.hours_left[car] = next_obs_time_left[car]
                    self.episode.old_soc[car] = self.episode.soc[car]
                    self.episode.soc[car] = next_obs_soc[car]

                # this shouldn't happen but if it does, an error is thrown
                else:
                    raise TypeError("Observation format not recognized")

                # if battery degradation >= 10%, target SOC is increased to ensure sufficient kWh in the battery
                if self.episode.soh[car] <= 0.9:
                    self.target_soc[car] = 0.9
                    self.episode.events += 1  # relevant event detected
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
                self.episode.events += 1  # relevant event detected
                if self.calc_deg:
                    self.deg_data_logger.add_log_entry()
                if self.print_updates:
                    print(f"Episode done: {self.episode.done}")
                    self.logged_data = self.data_logger.log

            # append to the reward history
            self.episode.cumulative_reward += reward
            self.episode.reward_history.append((self.episode.time, self.episode.cumulative_reward))

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
            penalty = reward - (cashflow * self.score_config.price_multiplier)
            grid = abs(overload_amount)
            soc_v = abs(cum_soc_missing)

            # Calculate degradation and state of health based on chosen method
            # calculate degradation once per day
            if self.calc_deg and ((self.episode.time.hour == 14) and (self.episode.time.minute == 45)):
                degradation = self.sei_deg.calculate_degradation(self.deg_data_logger.soc_log,
                                                                 self.load_calculation.evse_max_power,
                                                                 self.time_conf,
                                                                 self.ev_config.temperature)
                # calculate SOH from current degradation
                self.episode.soh = np.subtract(self.episode.soh, degradation)
                # calculate new resulting battery capacity after degradation
                self.episode.battery_cap = self.episode.soh * self.ev_config.init_battery_cap
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

            if not self.real_time:
                break

            if self.event_manager.check_event(self.episode):
                if self.print_updates:
                    print("Relevant event recognised. Will pass to RL agent.")
                self.episode.events = 0
                break

        # return according to openAI gym core API
        return norm_next_obs, reward, self.episode.done, False, self.info

    def close(self):
        return None

    def print(self, action):
        """
        The print function can provide useful information of the environment dynamics and the agent's actions.
        Can slow down FPS due to the printing at each timestep

        :param action: Action of the agent
        :return: None -> Just prints information if specified
        """
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
        if self.render_mode == "human":
            there = self.db["There"][self.db["date"] == self.episode.time].values
            kw = np.multiply(self.episode.current_actions, self.load_calculation.evse_max_power)
            soc = self.episode.soc
            if there is None:
                there = np.zeros(self.num_cars)
                kw = np.zeros(self.num_cars)
                soc = np.zeros(self.num_cars)
            self.pl_render.render(there=there, kw = kw, soc = soc)

    # functions that can be called through vec_envs via env_method()
    def get_log(self):
        """
        This function can be called through SB3 vectorized environments via VecEnv.env_method("get_log")[0]
        The zero index is required so only the first element -> the DataFrame is extracted

        :return: Log dataframe
        """
        return self.data_logger.log

    def is_done(self):
        """
        VecEnv.env_method("is_done")[0]
        :return: Flag is episode is done, bool
        """
        # return if episode is done
        return self.episode.done

    def get_start_time(self):
        """
        VecEnv.env_method("get_start_time")[0]
        :return: pd.TimeStamp
        """
        return self.episode.start_time

    def set_start_time(self, start_time: str):
        """
        VecEnv.env_method("set_start_time", [f"{start_time}"])
        Must parse the function and argument of start_time
        :param start_time: string of pd.TimeStamp / date
        :return: None
        """
        self.episode.start_time = start_time
        return None

    def get_time(self):
        """
        VecEnv.env_method("get_time")[0]
        :return: pd.TimeStamp: current timestamp
        """
        return self.episode.time

    def get_dist_factor(self):
        """
        This function returns the distribution/laxity factor: how much time needed vs. how much time left at charger
        If factor is 0.1, the dist agent would only charge with 10% of the EVSE capacity.
        Call via env_method("get_dist_factor")[0] if using an SB3 Vectorized Environment
        :return: dist/laxity factor, float
        """

        obs = self.observer.get_obs(self.db,
                                    self.time_conf.price_lookahead,
                                    self.time_conf.bl_pv_lookahead,
                                    self.episode.time,
                                    ev_conf=self.ev_config,
                                    load_calc=self.load_calculation,
                                    aux=self.aux_flag,
                                    target_soc=self.target_soc)

        return np.divide(obs["hours_needed"], np.add(obs["hours_left"], 0.001))

    def choose_time_picker(self, time_picker):
        """
        Chooses the right time picker based on the specified in input string.
        Static: Always the same time is picked to start an episode
        Random: Start an episode randomly from the training set
        Eval: Start an episode randomly from the validation set
        :param time_picker: (string), specifies which time picker to choose: "static", "eval", "random"
        :return: tp (TimePicker) -> time picker object
        """

        # Load time picker module
        if time_picker == "static":
            # when an episode starts, this class picks the same starting time
            tp: TimePicker = StaticTimePicker()
        elif time_picker == "eval":
            # picks a random starting times from test set (nov - dez)
            tp: TimePicker = EvalTimePicker(self.time_conf.episode_length)
        elif time_picker == "random":
            # picks random starting times from training set (jan - oct)
            tp: TimePicker = RandomTimePicker()
        else:
            # must choose between static, eval or random
            raise TypeError("Time picker type not recognised")

        return tp

    def choose_observer(self):
        """
        This function chooses the right observer, depending on whether to include price, building, PV, etc.
        :return: obs (Observer) -> The observer module to choose
        """

        # All observations are made in the observer class
        # not even price: only soc and time left
        if not self.include_price:
            obs: Observer = ObserverSocTimeOnly()
        # only price
        elif not self.include_building_load and not self.include_pv:
            obs: Observer = ObserverPriceOnly()
        # price and building load
        elif self.include_building_load and not self.include_pv:
            obs: Observer = ObserverWithBuildingLoad()
        # price and pv
        elif not self.include_building_load and self.include_pv:
            obs: Observer = ObserverWithPV()
        # price, building load and pv
        elif self.include_building_load and self.include_pv:
            obs: Observer = ObserverWithBoth()
        else:
            raise TypeError("Observer configuration not found. Recheck flags.")

        return obs

    def detect_dim_and_bounds(self):

        """
        This function chooses the right dimension of the observation space based on the chosen configuration.
        Each increase of dim is explained below. The low_obs and high_obs are built in the normalizer object,
        using the dim value that was calculated in this function.

        - set boundaries of the observation space, detects if normalized or not.
        - If aux flag is true, additional information enlarges the observation space.
        - The following code goes through all possible environment setups.
        - Depending on the setup, the dimensions differ and every case is handled differently.

        :return: low_obs and high_obs: tuple[float, float] | tuple[np.ndarray, np.ndarray] -> used for gym.Spaces
        """

        if not self.include_price:
            dim = 2 * self.num_cars  # soc and time left for each EV
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

        return low_obs, high_obs

    def adjust_caretaker_lunch_soc(self):
        """
        The caretaker target SOC can be set lower during the lunch break to avoid unfair penalties occurring. This is
        because the break is not long enough to charge until 0.85 target SOC.
        :return: None -> sets the target SOC during lunch break hours to 0.65 by default
        """
        # make an adjustment for caretakers: the afternoon tour SOC on arrival should be calculated with the
        # afternoon target SOC. This is set to 0.65 in this case
        afternoon_trips = self.db.loc[((self.db["date"].dt.hour >= 0) & (self.db["date"].dt.hour <= 10))
                                      | ((self.db["date"].dt.hour >= 15) & (self.db["date"].dt.hour <= 23))]

        self.db.loc[((self.db["date"].dt.hour >= 0) & (self.db["date"].dt.hour <= 10))
                    | ((self.db["date"].dt.hour >= 15) & (self.db["date"].dt.hour <= 23)), "SOC_on_return"] \
            = (self.ev_config.target_soc_lunch
               - afternoon_trips["last_trip_total_consumption"].div(self.ev_config.init_battery_cap))

        self.db.loc[self.db["There"] == 0, "SOC_on_return"] = 0

    def auto_gen(self):
        """
        This function automatically generates schedules as specified.
        Uses the ScheduleGenerator module.
        Note: this can take up to 1-3 hours, depending on the number of vehicles.

        :return: None -> The schedule is generated and placed in the input folder
        """
        gen_sched = []

        print("Generating schedules... This may take a while.")
        for i in range(self.gen_n_evs):
            self.schedule_gen = ScheduleGenerator(env_config=self.env_config,
                                                  schedule_type=self.schedule_type,
                                                  vehicle_id=str(i))

            gen_sched.append(self.schedule_gen.generate_schedule())

        complete_schedule = pd.concat(gen_sched)
        if not self.gen_name.endswith(".csv"):
            self.gen_name = self.gen_name + ".csv"
        complete_schedule.to_csv(os.path.join(self.path_name, self.gen_name))
        print(f"Schedule generation complete. Saved in Inputs path. File name: {self.gen_name}")
        self.schedule_name = self.gen_name

    def get_next_dt(self):

        """
        Calculates the time delta from the current time step and the next one. This allows for csv input files that
        have irregular time intervals. Energy calculations will automatically adjust for the dynamic time differences
        through kWh = kW * dt

        :return: next time delta in hours
        """

        current_time = self.episode.time
        next_time = self.db["date"][self.db["date"].searchsorted(current_time) + 1]
        delta = (next_time - current_time).total_seconds()/3600
        return delta

    def get_next_minutes(self):

        """
        Calculates the integer of minutes until the next time step. This therefore limits the framework's current
        maximum resolution to discrete time steps of 1 min. This will be improved soon, as well as the dependency to
        know the future value beforehand.

        :return: Integer of minutes until next timestep
        """

        current_time = self.episode.time
        next_time = self.db["date"][self.db["date"].searchsorted(current_time) + 1]
        delta = (next_time - current_time).total_seconds()/60
        return int(delta)

    def read_config(self, conf_path: str):

        with open(f'{conf_path}', 'r') as file:
            env_config = json.load(file)
            return env_config

    def check_data_paths(self, input_path, schedule_path, spot_path, load_path, pv_path):

        schedule = os.path.join(input_path, schedule_path) if schedule_path is not None else None
        spot = os.path.join(input_path, spot_path) if spot_path is not None else None
        load = os.path.join(input_path, load_path) if load_path is not None else None
        pv = os.path.join(input_path, pv_path) if pv_path is not None else None

        for path in [schedule, spot, load, pv]:
            if path is not None:
                assert(os.path.isfile(path)), f"Path does not exist: {path}"

    def specify_company_and_battery_size(self, use_case):
        # Specify company type and associated battery size in kWh
        if use_case == "ct":
            self.company = CompanyType.Caretaker
            self.schedule_type = ScheduleType.Caretaker
            self.ev_config.init_battery_cap = 16.7
        elif use_case == "ut":
            self.company = CompanyType.Utility
            self.schedule_type = ScheduleType.Utility
            self.ev_config.init_battery_cap = 50.0
        elif use_case == "lmd":
            self.company = CompanyType.Delivery
            self.schedule_type = ScheduleType.Delivery
            self.ev_config.init_battery_cap = 60.0
        elif use_case == "custom":
            self.company = CompanyType.Custom
            self.schedule_type = ScheduleType.Custom
            self.ev_config.init_battery_cap = self.env_config["custom_ev_battery_size_in_kwh"]
        else:
            raise TypeError("Company not recognised.")

    def change_markups(self):
        if self.env_config["spot_markup"] is not None:
            self.ev_config.fixed_markup = self.env_config["spot_markup"]
        if self.env_config["spot_mul"] is not None:
            self.ev_config.variable_multiplier = self.env_config["spot_mul"]
        if self.env_config["feed_in_ded"] is not None:
            self.ev_config.feed_in_deduction = self.env_config["feed_in_ded"]

    def adjust_score_config(self):
        if self.env_config["ignore_price_reward"]:
            self.score_config.price_multiplier = 0
        if self.env_config["ignore_overloading_penalty"]:
            self.score_config.penalty_overloading = 0
        if self.env_config["ignore_invalid_penalty"]:
            self.score_config.penalty_invalid_action = 0
        if self.env_config["ignore_overcharging_penalty"]:
            self.score_config.penalty_overcharging = 0

