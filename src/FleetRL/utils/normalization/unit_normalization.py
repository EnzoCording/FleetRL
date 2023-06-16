import numpy as np

from FleetRL.utils.normalization.normalization import Normalization


# No normalization, just concatenation and properly adjusting the boundaries
# NB: Also uses global max and min values that might not be known in a real case scenario for future time steps
class UnitNormalization(Normalization):

    def __init__(self, db, num_cars, price_lookahead, bl_pv_lookahead, timestep_per_hour, building_flag, pv_flag, price_flag):
        self.max_time_left = max(db["time_left"])
        self.max_spot = max(db["DELU"])
        self.min_spot = min(db["DELU"])
        if building_flag:
            self.min_building_load = min(db["load"])
            self.max_building_load = max(db["load"])
        if pv_flag:
            self.max_pv = max(db["pv"])

        if not price_flag:
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars)
                 ),
                dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 ),
                dtype=np.float32)

        if (not building_flag) and (not pv_flag):

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

        if (building_flag) and (not pv_flag):
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.min_spot),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.min_building_load)
                 ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.max_building_load)
                 ), dtype=np.float32)

        if (not building_flag) and (pv_flag):
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.min_spot),
                 np.zeros(bl_pv_lookahead * timestep_per_hour)
                 ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.max_pv)
                 ), dtype=np.float32)

        if building_flag and pv_flag:
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.min_spot),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.min_building_load),
                 np.zeros(bl_pv_lookahead * timestep_per_hour)
                 ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead * timestep_per_hour, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.max_building_load),
                 np.full(shape=bl_pv_lookahead * timestep_per_hour, fill_value=self.max_pv)
                 ), dtype=np.float32)

    def normalize_obs(self, obs: list) -> np.ndarray:
        return np.concatenate(obs, dtype=np.float32)

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        return self.low_obs, self.high_obs
