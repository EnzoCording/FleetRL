import numpy as np
import pandas as pd

from FleetRL.utils.normalization.unit_normalization import Normalization
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

# This normalizes based on the global maximum values. These could be in the future, hence the oracle prefix.
# For more realistic approaches, a rolling average could be used, or the sb3 vec normalize function
class OracleNormalization(Normalization):
    def __init__(self,
                 db,
                 building_flag,
                 pv_flag,
                 price_flag,
                 ev_conf: EvConfig,
                 load_calc: LoadCalculation, aux: bool):

        self.max_time_left = max(db["time_left"])
        self.max_price = (max(db["DELU"]) + ev_conf.fixed_markup) * ev_conf.variable_multiplier
        self.min_price = (min(db["DELU"]) + ev_conf.fixed_markup) * ev_conf.variable_multiplier
        self.max_tariff = (max(db["tariff"])) * (1 - ev_conf.feed_in_deduction)
        self.min_tariff = (min(db["tariff"])) * (1 - ev_conf.feed_in_deduction)
        self.building_flag = building_flag
        self.pv_flag = pv_flag
        self.price_flag = price_flag
        self.aux = aux

        if self.building_flag:
            self.max_building = max(db["load"])
        if self.pv_flag:
            self.max_pv = max(db["pv"])
        if self.aux:
            self.max_soc = ev_conf.target_soc
            self.max_hours_needed = (ev_conf.target_soc * ev_conf.init_battery_cap)/(load_calc.evse_max_power * ev_conf.charging_eff)
            self.max_laxity = 5
            self.max_evse = load_calc.evse_max_power
            self.max_grid = load_calc.grid_connection

    def normalize_obs(self, input_obs: dict) -> np.ndarray:
        # normalization is done here, so if the rule is changed it is automatically adjusted in step and reset
        input_obs["soc"] = (input_obs["soc"])  # soc is already normalized
        input_obs["hours_left"] = list(np.divide(input_obs["hours_left"], self.max_time_left))  # max hours of entire db
        # normalize spot price between 0 and 1, there are negative values
        # formula: z_i = (x_i - min(x))/(max(x) - min(x))
        if self.price_flag:
            input_obs["price"] = list(np.divide(np.subtract(input_obs["price"], self.min_price),
                                                np.subtract(self.max_price, self.min_price)))
            input_obs["tariff"] = list(np.divide(np.subtract(input_obs["tariff"], self.min_tariff),
                                                 np.subtract(self.max_tariff, self.min_tariff)))

        if not self.price_flag:
            output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)
            if self.aux:
                input_obs["there"] = list(np.divide(input_obs["there"], 1))  # there
                input_obs["target_soc"] = list(np.divide(input_obs["target_soc"], self.max_soc))  # target soc
                input_obs["charging_left"] = list(np.divide(input_obs["charging_left"], self.max_soc))  # charging left
                input_obs["hours_needed"] = list(np.divide(input_obs["hours_needed"], self.max_hours_needed))  # hours needed
                input_obs["laxity"] = list(np.divide(input_obs["laxity"], self.max_laxity))  # laxity
                input_obs["evse_power"] = list(np.divide(input_obs["evse_power"], self.max_evse))  # evse power
                output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)

        elif not self.building_flag and not self.pv_flag:
            output_obs = np.concatenate(
                (input_obs["soc"], input_obs["hours_left"], input_obs["price"],
                 input_obs["tariff"]), dtype=np.float32)
            if self.aux:
                input_obs["there"] = list(np.divide(input_obs["there"], 1))  # there
                input_obs["target_soc"] = list(np.divide(input_obs["target_soc"], self.max_soc))  # target soc
                input_obs["charging_left"] = list(np.divide(input_obs["charging_left"], self.max_soc))  # charging left
                input_obs["hours_needed"] = list(np.divide(input_obs["hours_needed"], self.max_hours_needed))  # hours needed
                input_obs["laxity"] = list(np.divide(input_obs["laxity"], self.max_laxity))  # laxity
                input_obs["evse_power"] = list(np.divide(input_obs["evse_power"], self.max_evse))  # evse power
                # input obs of month, week and hour sin/cos don't need to be normalized
                output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)

        elif self.building_flag and not self.pv_flag:
            input_obs["building_load"] = list(np.divide(input_obs["building_load"], self.max_building))
            output_obs = np.concatenate(
                (input_obs["soc"], input_obs["hours_left"], input_obs["price"],
                 input_obs["tariff"], input_obs["building_load"]
                 ), dtype=np.float32)

            if self.aux:
                input_obs["there"] = list(np.divide(input_obs["there"], 1))  # there
                input_obs["target_soc"] = list(np.divide(input_obs["target_soc"], self.max_soc))  # target soc
                input_obs["charging_left"] = list(np.divide(input_obs["charging_left"], self.max_soc))  # charging left
                input_obs["hours_needed"] = list(np.divide(input_obs["hours_needed"], self.max_hours_needed))  # hours needed
                input_obs["laxity"] = list(np.divide(input_obs["laxity"], self.max_laxity))  # laxity
                input_obs["evse_power"] = list(np.divide(input_obs["evse_power"], self.max_evse))  # evse power
                input_obs["grid_cap"] = list(np.divide(input_obs["grid_cap"], self.max_grid))  # grid connection
                input_obs["avail_grid_cap"] = list(np.divide(input_obs["avail_grid_cap"], self.max_grid))  # available grid
                input_obs["possible_avg_action"] = list(np.divide(input_obs["possible_avg_action"], 1))  # possible avg action per car
                # input obs of month, week and hour sin/cos don't need to be normalized
                output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)

        elif not self.building_flag and self.pv_flag:
            input_obs["building_load"] = list(np.divide(input_obs["building_load"], self.max_building))
            output_obs = np.concatenate(
                (input_obs["soc"], input_obs["hours_left"], input_obs["price"],
                 input_obs["tariff"], input_obs["pv"]
                 ), dtype=np.float32)

            if self.aux:
                input_obs["there"] = list(np.divide(input_obs["there"], 1))  # there
                input_obs["target_soc"] = list(np.divide(input_obs["target_soc"], self.max_soc))  # target soc
                input_obs["charging_left"] = list(np.divide(input_obs["charging_left"], self.max_soc))  # charging left
                input_obs["hours_needed"] = list(np.divide(input_obs["hours_needed"], self.max_hours_needed))  # hours needed
                input_obs["laxity"] = list(np.divide(input_obs["laxity"], self.max_laxity))  # laxity
                input_obs["evse_power"] = list(np.divide(input_obs["evse_power"], self.max_evse))  # evse power
                # input obs of month, week and hour sin/cos don't need to be normalized
                output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)

        elif self.building_flag and self.pv_flag:
            input_obs["building_load"] = list(np.divide(input_obs["building_load"], self.max_building))
            input_obs["pv"] = list(np.divide(input_obs["pv"], self.max_pv))
            output_obs = np.concatenate(
                (input_obs["soc"], input_obs["hours_left"], input_obs["price"],
                 input_obs["tariff"], input_obs["building_load"], input_obs["pv"]
                 ), dtype=np.float32)

            if self.aux:
                input_obs["there"] = list(np.divide(input_obs["there"], 1))  # there
                input_obs["target_soc"] = list(np.divide(input_obs["target_soc"], self.max_soc))  # target soc
                input_obs["charging_left"] = list(np.divide(input_obs["charging_left"], self.max_soc))  # charging left
                input_obs["hours_needed"] = list(np.divide(input_obs["hours_needed"], self.max_hours_needed))  # hours needed
                input_obs["laxity"] = list(np.divide(input_obs["laxity"], self.max_laxity))  # laxity
                input_obs["evse_power"] = list(np.divide(input_obs["evse_power"], self.max_evse))  # evse power
                input_obs["grid_cap"] = list(np.divide(input_obs["grid_cap"], self.max_grid))  # grid connection
                input_obs["avail_grid_cap"] = list(np.divide(input_obs["avail_grid_cap"], self.max_grid))  # available grid
                input_obs["possible_avg_action"] = list(np.divide(input_obs["possible_avg_action"], 1))  # possible avg action per car
                # input obs of month, week and hour sin/cos don't need to be normalized
                output_obs = np.array(self.flatten_obs(input_obs), dtype=np.float32)

        else:
            output_obs = None
            raise RuntimeError("Problem with included information. Check building and PV flags.")

        return output_obs

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        low_obs = np.zeros(dim, dtype=np.float32)
        high_obs = np.ones(dim, dtype=np.float32)
        return low_obs, high_obs

    @staticmethod
    def flatten_obs(obs):
        flattened_obs = [v if isinstance(v, list) else [v] for v in obs.values()]
        flattened_obs = [item for sublist in flattened_obs for item in sublist]
        return flattened_obs
