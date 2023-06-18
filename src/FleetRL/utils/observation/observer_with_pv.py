import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer


class ObserverWithPV(Observer):
    def get_obs(self, db: pd.DataFrame, price_lookahead: int, bl_pv_lookahead:int, time: pd.Timestamp) -> list:

        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        price=pd.DataFrame()
        pv = pd.DataFrame()

        price_start = np.where(db["date"] == time)[0][0]
        price_end = np.where(db["date"] == (time + np.timedelta64(price_lookahead+2, 'h')))[0][0]
        price["DELU"] = db["DELU"][price_start: price_end]
        price["date"] = db["date"][price_start: price_end]
        price = price.resample("H", on="date").first()["DELU"].values
        price = price[0:price_lookahead+1]

        pv_start = np.where(db["date"] == time)[0][0]
        pv_end = np.where(db["date"] == (time + np.timedelta64(bl_pv_lookahead+2, 'h')))[0][0]
        pv["pv"] = db["pv"][pv_start: pv_end]
        pv["date"] = db["date"][pv_start: pv_end]
        pv = pv.resample("H", on="date").first()["pv"].values
        pv = pv[0:bl_pv_lookahead+1]

        return [soc, hours_left, price, pv]

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