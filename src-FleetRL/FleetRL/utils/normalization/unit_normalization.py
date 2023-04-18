import numpy as np

from FleetRL.utils.normalization.normalization import Normalization


class UnitNormalization(Normalization):

    def __init__(self, db, spot_price, num_cars, price_lookahead, timestep_per_hour):
        self.max_time_left = max(db["time_left"])
        self.max_spot = max(spot_price["DELU"])
        self.min_spot = min(spot_price["DELU"])

        self.low_obs = np.concatenate(
            (np.zeros(num_cars),
             np.zeros(num_cars),
             np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.min_spot)
             ), dtype=np.float32)

        self.high_obs = np.concatenate(
            (np.ones(num_cars),
             np.full(shape=num_cars, fill_value=self.max_time_left),
             np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.max_spot)
             ), dtype=np.float32)

    def normalize_obs(self, obs: list) -> np.ndarray:
        return np.concatenate(obs, dtype=np.float32)

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        return self.low_obs, self.high_obs
