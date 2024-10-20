import numpy as np
import pandas as pd
from fleetrl.utils.load_calculation.load_calculation import LoadCalculation
from fleetrl.fleet_env.config.ev_config import EvConfig
from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob


class Observer:
    """
    Parent class for observer modules.
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

        raise NotImplementedError("This is an abstract class")

    # Always the same, so can be defined in base class
    @staticmethod
    def get_trip_len(db: pd.DataFrame, car: int, time: pd.Timestamp) -> float:
        """
        :param db: from the env
        :param car: car ID
        :param time: current timestamp
        :return: length of trip in hours as a float
        """
        raise NotImplementedError("This is an abstract class")
