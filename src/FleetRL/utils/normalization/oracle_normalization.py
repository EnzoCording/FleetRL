import numpy as np

from FleetRL.utils.normalization.unit_normalization import Normalization
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

# This normalizes based on the global maximum values. These could be in the future, hence the oracle prefix.
# For more realistic approaches, a rolling average could be used, or the sb3 vec normalize function
class OracleNormalization(Normalization):
    def __init__(self, db, building_flag, pv_flag, price_flag,
                 ev_conf: EvConfig, load_calc: LoadCalculation, aux: bool):
        self.max_time_left = max(db["time_left"])
        self.max_spot = max(db["DELU"])
        self.min_spot = min(db["DELU"])
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
            self.max_hours_needed = (ev_conf.target_soc * ev_conf.battery_cap)\
                                    / (load_calc.evse_max_power * ev_conf.charging_eff)
            self.max_laxity = 5
            self.max_evse = load_calc.evse_max_power
            self.max_grid = load_calc.grid_connection

    def normalize_obs(self, input_obs: list) -> np.ndarray:
        # normalization is done here, so if the rule is changed it is automatically adjusted in step and reset
        input_obs[0] = np.array(input_obs[0])  # soc is already normalized
        input_obs[1] = np.array(input_obs[1] / self.max_time_left)  # max hours of entire db
        # normalize spot price between 0 and 1, there are negative values
        # formula: z_i = (x_i - min(x)) / (max(x) - min(x))
        if self.price_flag:
            input_obs[2] = np.array((input_obs[2] - self.min_spot)
                                    / (self.max_spot - self.min_spot))
        if not self.price_flag:
            output_obs = np.concatenate((input_obs[0], input_obs[1]), dtype=np.float32)
        elif not self.building_flag and not self.pv_flag:
            output_obs = np.concatenate((input_obs[0], input_obs[1], input_obs[2]), dtype=np.float32)
        elif self.building_flag and not self.pv_flag:
            input_obs[3] = np.array(input_obs[3] / self.max_building)
            output_obs = np.concatenate((input_obs[0], input_obs[1], input_obs[2], input_obs[3]), dtype=np.float32)
        elif not self.building_flag and self.pv_flag:
            input_obs[3] = np.array(input_obs[3] / self.max_pv)
            output_obs = np.concatenate((input_obs[0], input_obs[1], input_obs[2], input_obs[3]), dtype=np.float32)
        elif self.building_flag and self.pv_flag:
            input_obs[3] = np.array(input_obs[3] / self.max_building)
            input_obs[4] = np.array(input_obs[4] / self.max_pv)

            output_obs = np.concatenate(
                (input_obs[0], input_obs[1], input_obs[2], input_obs[3], input_obs[4]
                 ), dtype=np.float32)

            if self.aux:
                input_obs[5] = np.array(input_obs[5] / 1)  # there
                input_obs[6] = np.array(input_obs[6] / self.max_soc)  # target soc
                input_obs[7] = np.array(input_obs[7] / self.max_soc)  # charging left
                input_obs[8] = np.array(input_obs[8] / self.max_hours_needed)  # hours needed
                input_obs[9] = np.array(input_obs[9] / self.max_laxity)  # laxity
                input_obs[10] = np.array(input_obs[10] / self.max_evse)  # evse power
                input_obs[11] = np.array(input_obs[11] / self.max_grid)  # grid connection
                input_obs[12] = np.array(input_obs[12] / self.max_grid)  # available grid
                input_obs[13] = np.array(input_obs[13] / 1)  # possible avg action per car

                output_obs = np.concatenate(
                    (input_obs[0], input_obs[1], input_obs[2], input_obs[3], input_obs[4],
                     input_obs[5], input_obs[6], input_obs[7], input_obs[8], input_obs[9],
                     input_obs[10], input_obs[11], input_obs[12], input_obs[13]
                     ), dtype=np.float32)

        else:
            output_obs = None
            raise RuntimeError("Problem with included information. Check building and PV flags.")

        return output_obs

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        low_obs = np.zeros(dim, dtype=np.float32)
        high_obs = np.ones(dim, dtype=np.float32)
        return low_obs, high_obs
