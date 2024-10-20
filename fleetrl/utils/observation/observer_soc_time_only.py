import numpy as np
import pandas as pd

from fleetrl.utils.observation.observer import Observer
from fleetrl.utils.load_calculation.load_calculation import LoadCalculation
from fleetrl.fleet_env.config.ev_config import EvConfig
from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob


class ObserverSocTimeOnly(Observer):

    """
    Observer that only regards SOC and time left, but not charging cost, PV or building laod
    """

    def get_obs(self,
                db: pd.DataFrame,
                price_lookahead: int,
                bl_pv_lookahead:int,
                time: pd.Timestamp,
                charging_efficiency: float,
                variable_multiplier: float,
                fixed_markup: float,
                feed_in_deduction: float,
                battery_capacity: float,
                max_charger_power: float,
                grid_connection: float,
                aux: bool,
                target_soc: list) -> dict:

        """
        - define the starting and ending time via lookahead, np.where returns the index in the dataframe
        - add lookahead + 2 here because of rounding issues with the resample function on square times (00:00)
        - get data of price and date from the specific indices
        - resample data to only include one value per hour (the others are duplicates)
        - only take into account the current value, and the specified hours of lookahead

        :param grid_connection:
        :param max_charger_power:
        :param feed_in_deduction:
        :param variable_multiplier:
        :param fixed_markup:
        :param battery_capacity: Energy capacity in kWh
        :param charging_efficiency: Charging efficiency
        :param db: database from the env
        :param price_lookahead: lookahead window for spot price
        :param bl_pv_lookahead: lookahead window for building load and pv
        :param time: current time of time step
        :param ev_conf: EV config needed for batt capacity and other params
        :param load_calc: Load calc module needed for grid connection and other params
        :param aux: Include auxiliary information that might help the agent to learn the problem
        :param target_soc: A list of target soc values, one for each car
        :return: Returns a list of np arrays that make up different parts of the observation.
        """

        # soc and time left always present in environment
        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        ###
        # Auxiliary observations that might make it easier for the agent
        # target soc
        there = db.loc[db["date"] == time, "There"].values
        target_soc = target_soc * there
        # maybe need to typecast to list
        charging_left = np.subtract(target_soc, soc)
        hours_needed = charging_left * battery_capacity / (max_charger_power * charging_efficiency)
        laxity = np.subtract(hours_left / (np.add(hours_needed, 0.001)), 1) * there
        laxity = np.clip(laxity, 0, 5)
        # could also be a vector
        evse_power = max_charger_power * np.ones(1)

        month_sin = np.sin(2 * np.pi * time.month / 12)
        month_cos = np.cos(2 * np.pi * time.month / 12)

        week_sin = np.sin(2 * np.pi * time.weekday() / 7)
        week_cos = np.cos(2 * np.pi * time.weekday() / 7)

        hour_sin = np.sin(2 * np.pi * time.hour / 24)
        hour_cos = np.cos(2 * np.pi * time.hour / 24)

        obs = {
            "soc": list(soc),  # state of charge
            "hours_left": list(hours_left),  # hours left at the charger
            "there": list(there),  # boolean, is the car i there or not
            "target_soc": list(target_soc),  # target soc of car i
            "charging_left": list(charging_left),  # charging % left
            "hours_needed": list(hours_needed),  # hours needed to get to target soc
            "laxity": list(laxity),  # laxity factor
            "evse_power": list(evse_power),  # evse power in kW
            "month_sin": month_sin,  # month in sin, and so on
            "month_cos": month_cos,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos
        }

        if aux:
            return obs
        else:
            return {key: obs[key] for key in ["soc", "hours_left"]}

    @staticmethod
    def get_trip_len(db: pd.DataFrame, car: int, time: pd.Timestamp) -> float:
        """
        :param db: from the env
        :param car: car ID
        :param time: current timestamp
        :return: length of trip in hours as a float
        """

        trip_len = db.loc[(db["ID"] == car) & (db["date"] == time), "last_trip_total_length_hours"].values

        return trip_len
