import numpy as np
import pandas as pd
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

class Observer:
    def get_obs(self,
                db: pd.DataFrame,
                price_lookahead: int,
                bl_pv_lookahead:int,
                time: pd.Timestamp,
                ev_conf: EvConfig,
                load_calc: LoadCalculation,
                aux: bool,
                target_soc: list) -> dict:

        """
        :param db: database from the env
        :param price_lookahead: lookahead window for spot price
        :param bl_pv_lookahead: lookahead window for building load and pv
        :param time: current time of time step
        :param ev_conf: EV config needed for batt capacity and other params
        :param load_calc: Load calc module needed for grid connection and other params
        :param aux: Include auxiliary information that might help the agent to learn the problem
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

    @staticmethod
    def flatten_obs(obs):
        raise NotImplementedError("This is an abstract class")
