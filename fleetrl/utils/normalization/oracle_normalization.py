import numpy as np
import pandas as pd

from fleetrl.utils.normalization.unit_normalization import Normalization
from fleetrl.utils.load_calculation.load_calculation import LoadCalculation
from fleetrl_2.jobs.ev_config_job import EvConfigJob
from fleetrl_2.jobs.site_parameters_job import SiteParametersJob


# This normalizes based on the global maximum values. These could be in the future, hence the oracle prefix.
# For more realistic approaches, a rolling average could be used, or the sb3 vec normalize function
class OracleNormalization(Normalization):
    """
    Oracle Normalization assumes the knowledge of the max and min values of the dataset. This is necessary to perform
    a global min/max normalization. Alternatively, a rolling normalization could be implemented.
    """
    def __init__(self,
                 db: pd.DataFrame,
                 building_flag: bool,
                 pv_flag: bool,
                 price_flag: bool,
                 battery_capacity: float,
                 target_soc: float,
                 charging_efficiency: float,
                 feed_in_deduction: float,
                 variable_multiplier: float,
                 fixed_markup: float,
                 max_charger_power: float,
                 max_grid_connection: float,
                 aux: bool):
        """
        Initialize max and min values of the dataset, globally.

        :param db: Database dataframe
        :param building_flag: Include building load flag
        :param pv_flag: Include PV flag
        :param price_flag: Include price flag
        :param ev_conf: EV configuration object
        :param load_calc: Load calculation object
        :param aux: Flag whether to include auxiliary observations or not (bool)
        """

        self.max_time_left = max(db["time_left"])
        self.max_price = (max(db["DELU"]) + fixed_markup) * variable_multiplier
        self.min_price = (min(db["DELU"]) + fixed_markup) * variable_multiplier
        self.max_tariff = (max(db["tariff"])) * (1 - feed_in_deduction)
        self.min_tariff = (min(db["tariff"])) * (1 - feed_in_deduction)
        self.building_flag = building_flag
        self.pv_flag = pv_flag
        self.price_flag = price_flag
        self.aux = aux

        if self.building_flag:
            self.max_building = max(db["load"])
        if self.pv_flag:
            self.max_pv = max(db["pv"])
        if self.aux:
            self.max_soc = target_soc
            self.max_hours_needed = ((target_soc * battery_capacity)
                                     /(max_charger_power * charging_efficiency))
            self.max_laxity = 5
            self.max_evse = max_charger_power
            self.max_grid = max_grid_connection

    def normalize_obs(self, input_obs: dict) -> np.ndarray:
        """
        Normalization function. Different cases are checked depending on the flags of PV, load, price, and aux.
        Input observations are a dictionary with clear namings, to make further changes in the code easy and readable.

        :param input_obs: Input observation: Un-normalized observations as specified in the observer.
        :return: Normalized observation.
        """
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
        """
        The boundaries are 0 and 1 because the observations are min/max normalized.

        :param dim: Dimension of the observation depending on the flags
        :return: Low and high observation arrays for gym.Spaces.
        """
        low_obs = np.zeros(dim, dtype=np.float32)
        high_obs = np.ones(dim, dtype=np.float32)
        return low_obs, high_obs

    @staticmethod
    def flatten_obs(obs):
        """
        Observations must be flattened for openAI gym compatibility. The parsed observation must be a flat array and
        not a dictionary. The dictionary either includes float or array. The function removes the nesting.

        :param obs: Normalized observation dictionary
        :return: A flattened array - necessary for the RL algorithms to be in a 1-dim array e.g. [v_1, ..., v_N]
        """
        flattened_obs = [v if isinstance(v, list) else [v] for v in obs.values()]
        flattened_obs = [item for sublist in flattened_obs for item in sublist]
        return flattened_obs
