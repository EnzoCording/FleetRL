import numpy as np

from FleetRL.utils.normalization.normalization import Normalization
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

# No normalization, just concatenation and properly adjusting the boundaries
# NB: Also uses global max and min values that might not be known in a real case scenario for future time steps
class UnitNormalization(Normalization):

    def __init__(self, db, num_cars, price_lookahead, bl_pv_lookahead,
                 building_flag, pv_flag, price_flag, aux: bool,
                 ev_conf: EvConfig, load_calc: LoadCalculation):
        self.max_time_left = max(db["time_left"])
        self.max_spot = max(db["DELU"])
        self.min_spot = min(db["DELU"])

        if building_flag:
            self.min_building_load = min(db["load"])
            self.max_building_load = max(db["load"])
        if pv_flag:
            self.max_pv = max(db["pv"])

        # if price flag is false, building and pv will not be included
        if not price_flag:
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars)
                 ),
                dtype=np.float32)

            if aux:
                self.low_obs = np.concatenate(
                    (self.low_obs,
                     np.zeros(num_cars),  # there
                     np.zeros(num_cars),  # target soc
                     np.zeros(num_cars),  # charging left
                     np.zeros(num_cars),  # hours needed
                     np.zeros(num_cars),  # laxity
                     np.zeros(1),  # evse power
                     ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 ),
                dtype=np.float32)

            if aux:
                self.high_obs = np.concatenate(
                    (self.high_obs,
                     np.ones(num_cars),  # there,
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # target soc
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # charging left
                     # hours needed
                     np.full(shape=num_cars, fill_value=(ev_conf.target_soc * ev_conf.init_battery_cap)
                                                        / (load_calc.evse_max_power * ev_conf.charging_eff)),
                     np.full(shape=num_cars, fill_value=5),  # laxity
                     np.full(shape=1, fill_value=load_calc.evse_max_power),  # evse power
                     ), dtype=np.float32)

        # only price
        elif (not building_flag) and (not pv_flag):

            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead+1, fill_value=self.min_spot)
                 ), dtype=np.float32)

            if aux:
                self.low_obs = np.concatenate(
                    (self.low_obs,
                     np.zeros(num_cars),  # there
                     np.zeros(num_cars),  # target soc
                     np.zeros(num_cars),  # charging left
                     np.zeros(num_cars),  # hours needed
                     np.zeros(num_cars),  # laxity
                     np.zeros(1),  # evse power
                     ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead+1, fill_value=self.max_spot)
                 ), dtype=np.float32)

            if aux:
                self.high_obs = np.concatenate(
                    (self.high_obs,
                     np.ones(num_cars),  # there,
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # target soc
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # charging left
                     # hours needed
                     np.full(shape=num_cars, fill_value=(ev_conf.target_soc * ev_conf.init_battery_cap)
                                                        / (load_calc.evse_max_power * ev_conf.charging_eff)),
                     np.full(shape=num_cars, fill_value=5),  # laxity
                     np.full(shape=1, fill_value=load_calc.evse_max_power),  # evse power
                     ), dtype=np.float32)

        # only price and building
        elif (building_flag) and (not pv_flag):
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead+1, fill_value=self.min_spot),
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.min_building_load)
                 ), dtype=np.float32)

            if aux:
                self.low_obs = np.concatenate(
                    (self.low_obs,
                     np.zeros(num_cars),  # there
                     np.zeros(num_cars),  # target soc
                     np.zeros(num_cars),  # charging left
                     np.zeros(num_cars),  # hours needed
                     np.zeros(num_cars),  # laxity
                     np.zeros(1),  # evse power
                     np.zeros(1),  # grid connection
                     np.zeros(1),  # available grid capacity
                     np.zeros(1),  # possible avg action per car
                     ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead+1, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.max_building_load)
                 ), dtype=np.float32)

            if aux:
                self.high_obs = np.concatenate(
                    (self.high_obs,
                     np.ones(num_cars),  # there,
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # target soc
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # charging left
                     # hours needed
                     np.full(shape=num_cars, fill_value=(ev_conf.target_soc * ev_conf.init_battery_cap)
                                                        / (load_calc.evse_max_power * ev_conf.charging_eff)),
                     np.full(shape=num_cars, fill_value=5),  # laxity
                     np.full(shape=1, fill_value=load_calc.evse_max_power),  # evse power
                     np.full(shape=1, fill_value=load_calc.grid_connection),  # grid connection
                     np.full(shape=1, fill_value=load_calc.grid_connection),  # available grid connection
                     np.ones(1),  # possible avg action per car
                     ), dtype=np.float32)

        # only price and pv
        elif (not building_flag) and (pv_flag):
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),
                 np.zeros(num_cars),
                 np.full(shape=price_lookahead+1, fill_value=self.min_spot),
                 np.zeros(bl_pv_lookahead+1)
                 ), dtype=np.float32)

            if aux:
                self.low_obs = np.concatenate(
                    (self.low_obs,
                     np.zeros(num_cars),  # there
                     np.zeros(num_cars),  # target soc
                     np.zeros(num_cars),  # charging left
                     np.zeros(num_cars),  # hours needed
                     np.zeros(num_cars),  # laxity
                     np.zeros(1),  # evse power
                     ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead+1, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.max_pv)
                 ), dtype=np.float32)

            if aux:
                self.high_obs = np.concatenate(
                    (self.high_obs,
                     np.ones(num_cars),  # there,
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # target soc
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # charging left
                     # hours needed
                     np.full(shape=num_cars, fill_value=(ev_conf.target_soc * ev_conf.init_battery_cap)
                                                        / (load_calc.evse_max_power * ev_conf.charging_eff)),
                     np.full(shape=num_cars, fill_value=5),  # laxity
                     np.full(shape=1, fill_value=load_calc.evse_max_power),  # evse power
                     ), dtype=np.float32)

        # price, pv and building
        elif building_flag and pv_flag:
            self.low_obs = np.concatenate(
                (np.zeros(num_cars),  # soc
                 np.zeros(num_cars),  # hours left
                 np.full(shape=price_lookahead+1, fill_value=self.min_spot),  # spot
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.min_building_load),  # building
                 np.zeros(bl_pv_lookahead+1)  # pv
                 ), dtype=np.float32)

            if aux:
                self.low_obs = np.concatenate(
                    (self.low_obs,
                     np.zeros(num_cars),  # there
                     np.zeros(num_cars),  # target soc
                     np.zeros(num_cars),  # charging left
                     np.zeros(num_cars),  # hours needed
                     np.zeros(num_cars),  # laxity
                     np.zeros(1),  # evse power
                     np.zeros(1),  # grid connection
                     np.zeros(1),  # available grid capacity
                     np.zeros(1),  # possible avg action per car
                     ), dtype=np.float32)

            self.high_obs = np.concatenate(
                (np.ones(num_cars),
                 np.full(shape=num_cars, fill_value=self.max_time_left),
                 np.full(shape=price_lookahead+1, fill_value=self.max_spot),
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.max_building_load),
                 np.full(shape=bl_pv_lookahead+1, fill_value=self.max_pv)
                 ), dtype=np.float32)

            if aux:
                self.high_obs = np.concatenate(
                    (self.high_obs,
                     np.ones(num_cars),  # there,
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # target soc
                     np.full(shape=num_cars, fill_value=ev_conf.target_soc),  # charging left
                     # hours needed
                     np.full(shape=num_cars, fill_value=(ev_conf.target_soc * ev_conf.init_battery_cap)
                                                        / (load_calc.evse_max_power * ev_conf.charging_eff)),
                     np.full(shape=num_cars, fill_value=5),  # laxity
                     np.full(shape=1, fill_value=load_calc.evse_max_power),  # evse power
                     np.full(shape=1, fill_value=load_calc.grid_connection),  # grid connection
                     np.full(shape=1, fill_value=load_calc.grid_connection),  # available grid connection
                     np.ones(1),  # possible avg action per car
                     ), dtype=np.float32)

    def normalize_obs(self, obs: list) -> np.ndarray:
        return np.concatenate(obs, dtype=np.float32)

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        return np.full(shape=dim, fill_value=-np.inf), np.full(shape=dim, fill_value=np.inf)
