import numpy as np

from FleetRL.utils.normalization.unit_normalization import Normalization


class OracleNormalization(Normalization):
    def __init__(self, db, spot_price):
        self.max_time_left = max(db["time_left"])
        self.max_spot = max(spot_price["DELU"])
        self.min_spot = min(spot_price["DELU"])

    def normalize_obs(self, input_obs: list) -> np.ndarray:
        # observation is done here, so if the rule is changed it is automatically adjusted in step and reset
        input_obs[0] = np.array(input_obs[0])  # soc is already normalized
        input_obs[1] = np.array(input_obs[1] / self.max_time_left)  # max hours plugged in of entire db
        # normalize spot price between 0 and 1, there are negative values
        # z_i = (x_i - min(x)) / (max(x) - min(x))
        input_obs[2] = np.array((input_obs[2] - self.min_spot)
                                / (self.max_spot - self.min_spot))

        output_obs = np.concatenate((input_obs[0], input_obs[1], input_obs[2]), dtype=np.float32)

        return output_obs

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        low_obs = np.zeros(dim, dtype=np.float32)
        high_obs = np.ones(dim, dtype=np.float32)
        return low_obs, high_obs
