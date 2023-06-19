import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

class ObserverPriceOnly(Observer):
    def get_obs(self, db: pd.DataFrame, price_lookahead: int, bl_pv_lookahead: int,
                time: pd.Timestamp, ev_conf: EvConfig, load_calc: LoadCalculation, aux: bool) -> list:

        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        price = pd.DataFrame()

        price_start = np.where(db["date"] == time)[0][0]
        price_end = np.where(db["date"] == (time + np.timedelta64(price_lookahead+2, 'h')))[0][0]
        price["DELU"] = db["DELU"][price_start: price_end]
        price["date"] = db["date"][price_start: price_end]
        price = price.resample("H", on="date").first()["DELU"].values
        price = price[0:price_lookahead+1]

        ###
        # Auxiliary observations that might make it easier for the agent
        there = db.loc[db["date"]==time, "There"].values
        target_soc = ev_conf.target_soc * there
        charging_left = np.subtract(target_soc, soc)
        hours_needed = charging_left * load_calc.batt_cap / (load_calc.evse_max_power * ev_conf.charging_eff)
        laxity = np.subtract(hours_left / (np.add(hours_needed, 0.001)), 1) * there
        laxity = np.clip(laxity, 0, 5)
        # could also be a vector
        evse_power = load_calc.evse_max_power * np.ones(1)

        # maybe previous action
        # [a1, ..., aN], sum, resulting power, building, pv, grid, total, penalty, total energy, price, reward
        ###

        if aux:
            return [soc, hours_left, price,
                    there, target_soc, charging_left, hours_needed, laxity, evse_power]
        else:
            return [soc, hours_left, price]

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
