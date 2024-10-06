import numpy as np
import pandas as pd

from fleetrl.utils.observation.observer import Observer
from fleetrl.utils.load_calculation.load_calculation import LoadCalculation
from fleetrl.fleet_env.config.ev_config import EvConfig
from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob


class ObserverPriceOnly(Observer):
    """
    Observer for a model that only takes into account the price, but disregards PV and building load.
    """
    def get_obs(self,
                db: pd.DataFrame,
                price_lookahead: int,
                bl_pv_lookahead: int,
                time: pd.Timestamp,
                ev_conf: EvConfigJob,
                site_parameters: SiteParametersJob,
                load_calc: LoadCalculation,
                aux: bool,
                target_soc: list) -> dict:

        """
        - define the starting and ending time via lookahead, np.where returns the index in the dataframe
        - add lookahead + 2 here because of rounding issues with the resample function on square times (00:00)
        - get data of price and date from the specific indices
        - resample data to only include one value per hour (the others are duplicates)
        - only take into account the current value, and the specified hours of lookahead

        :param site_parameters:
        :param db: Database from env
        :param price_lookahead: Lookahead in hours for price
        :param bl_pv_lookahead: Lookahead in hours for PV and building
        :param time: Current time
        :param ev_conf: EV config data, used for battery capacity, etc.
        :param load_calc: Load calc module, used for grid connection, etc.
        :param aux: Flag to include extra information on the problem or not. Can help with training
        :param target_soc: List for the current target SOC of each car
        :return: Dict of lists with different parts of the observation
        """

        # soc and time left always present in environment
        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        # variables to hold observation data
        price = pd.DataFrame()
        tariff = pd.DataFrame()

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
        price = np.multiply(np.add(price[0:price_lookahead+1], site_parameters.fixed_markup), site_parameters.variable_multiplier)
        tariff = np.multiply(tariff[0:price_lookahead+1], 1-site_parameters.feed_in_deduction)

        ###
        # Auxiliary observations that might make it easier for the agent
        # target soc
        there = db.loc[db["date"]==time, "There"].values
        target_soc = target_soc * there
        # maybe need to typecast to list
        charging_left = np.subtract(target_soc, soc)
        hours_needed = charging_left * load_calc.batt_cap / (load_calc.evse_max_power * ev_conf.battery.charging_eff)
        laxity = np.subtract(hours_left / (np.add(hours_needed, 0.001)), 1) * there
        laxity = np.clip(laxity, 0, 5)
        # could also be a vector
        evse_power = load_calc.evse_max_power * np.ones(1)

        month_sin = np.sin(2 * np.pi * time.month/12)
        month_cos = np.cos(2 * np.pi * time.month/12)

        week_sin = np.sin(2 * np.pi * time.weekday() / 7)
        week_cos = np.cos(2 * np.pi * time.weekday() / 7)

        hour_sin = np.sin(2 * np.pi * time.hour / 24)
        hour_cos = np.cos(2 * np.pi * time.hour / 24)

        obs = {
            "soc": list(soc),  # state of charge
            "hours_left": list(hours_left),  # hours left at the charger
            "price": list(price),  # price for charging
            "tariff": list(tariff),  # price received when discharging
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
            return {key: obs[key] for key in ["soc", "hours_left", "price", "tariff"]}


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
