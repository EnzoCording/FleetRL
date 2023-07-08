import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

class ObserverWithBoth(Observer):
    def get_obs(self, db: pd.DataFrame,
                price_lookahead: int,
                bl_pv_lookahead:int,
                time: pd.Timestamp,
                ev_conf: EvConfig,
                load_calc: LoadCalculation,
                aux: bool,
                target_soc: list) -> list:

        """
        :param db: Database from env
        :param price_lookahead: Lookahead in hours for price
        :param bl_pv_lookahead: Lookahead in hours for PV and building
        :param time: Current time
        :param ev_conf: EV config data, used for battery capacity, etc.
        :param load_calc: Load calc module, used for grid connection, etc.
        :param aux: Flag to include extra information on the problem or not. Can help with training
        :param target_soc: List for the current target SOC of each car
        :return: List of numpy arrays with different parts of the observation

        # define the starting and ending time via lookahead, np.where returns the index in the dataframe
        # add lookahead + 2 here because of rounding issues with the resample function on square times (00:00)
        # get data of price and date from the specific indices
        # resample data to only include one value per hour (the others are duplicates)
        # only take into account the current value, and the specified hours of lookahead
        """

        # soc and time left always present in environment
        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        # variables to hold observation data
        price = pd.DataFrame()
        tariff = pd.DataFrame()
        building_load = pd.DataFrame()
        pv = pd.DataFrame()

        # define the starting and ending time via lookahead, np.where returns the index in the dataframe
        price_start = np.where(db["date"] == time)[0][0]
        # add lookahead + 2 here because of rounding issues with the resample function on square times (00:00)
        price_end = np.where(db["date"] == (time + np.timedelta64(price_lookahead+2, 'h')))[0][0]
        # get data of price and date from the specific indices
        price["DELU"] = db["DELU"][price_start: price_end]
        price["date"] = db["date"][price_start: price_end]
        tariff["tariff"] = db["tariff"][price_start: price_end]
        tariff["date"] = db["date"][price_start: price_end]
        # resample data to only include one value per hour (the others are duplicates)
        price = price.resample("H", on="date").first()["DELU"].values
        tariff = tariff.resample("H", on="date").first()["tariff"].values
        # only take into account the current value, and the specified hours of lookahead
        price = np.multiply(np.add(price[0:price_lookahead+1], ev_conf.fixed_markup), ev_conf.variable_multiplier)
        tariff = np.multiply(tariff[0:price_lookahead+1], 1-ev_conf.feed_in_deduction)

        bl_start = np.where(db["date"] == time)[0][0]
        bl_end = np.where(db["date"] == (time + np.timedelta64(bl_pv_lookahead+2, 'h')))[0][0]
        building_load["load"] = db["load"][bl_start: bl_end]
        building_load["date"] = db["date"][bl_start: bl_end]
        building_load = building_load.resample("H", on="date").first()["load"].values
        building_load = building_load[0:bl_pv_lookahead+1]

        pv_start = np.where(db["date"] == time)[0][0]
        pv_end = np.where(db["date"] == (time + np.timedelta64(bl_pv_lookahead+2, 'h')))[0][0]
        pv["pv"] = db["pv"][pv_start: pv_end]
        pv["date"] = db["date"][pv_start: pv_end]
        pv = pv.resample("H", on="date").first()["pv"].values
        pv = pv[0:bl_pv_lookahead+1]

        ###
        # Auxiliary observations that might make it easier for the agent
        # target soc
        there = db.loc[db["date"]==time, "There"].values
        target_soc = target_soc * there
        # maybe need to typecast to list
        charging_left = np.subtract(target_soc, soc)
        hours_needed = charging_left * load_calc.batt_cap / (load_calc.evse_max_power * ev_conf.charging_eff)
        laxity = np.subtract(hours_left / (np.add(hours_needed, 0.001)), 1) * there
        laxity = np.clip(laxity, 0, 5)
        # could also be a vector
        evse_power = load_calc.evse_max_power * np.ones(1)

        grid_cap = load_calc.grid_connection * np.ones(1)
        avail_grid_cap = (grid_cap - building_load[0] + pv[0]) * np.ones(1)
        num_cars = db["ID"].max() + 1
        possible_avg_action_per_car = min(avail_grid_cap / (num_cars * evse_power), 1) * np.ones(1)

        #previous action
        # [a1, ..., aN], sum, resulting power, building, pv, grid, total, penalty, total energy, price, reward
        ###

        if aux:
            return [soc, hours_left, price, tariff, building_load, pv,
                    there, target_soc, charging_left, hours_needed, laxity,
                    evse_power, grid_cap, avail_grid_cap, possible_avg_action_per_car]
        else:
            return [soc, hours_left, price, tariff, building_load, pv]

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