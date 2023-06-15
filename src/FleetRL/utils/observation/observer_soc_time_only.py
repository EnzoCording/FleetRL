import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer


class ObserverSocTimeOnly(Observer):
    def get_obs(self, db: pd.DataFrame, price_lookahead: int, bl_pv_lookahead: int,
                time: pd.Timestamp) -> list:

        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        return [soc, hours_left]

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
