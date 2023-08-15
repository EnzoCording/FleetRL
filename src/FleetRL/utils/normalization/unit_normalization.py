import numpy as np
import pandas as pd

from FleetRL.utils.normalization.normalization import Normalization
from FleetRL.utils.load_calculation.load_calculation import LoadCalculation
from FleetRL.fleet_env.config.ev_config import EvConfig

# No normalization, just concatenation and properly adjusting the boundaries
# NB: Also uses global max and min values that might not be known in a real case scenario for future time steps
class UnitNormalization(Normalization):

    def normalize_obs(self, obs: dict) -> np.ndarray:
        return np.array(self.flatten_obs(obs), dtype=np.float32)

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        return np.full(shape=dim, fill_value=-np.inf), np.full(shape=dim, fill_value=np.inf)

    @staticmethod
    def flatten_obs(obs):
        flattened_obs = [v if isinstance(v, list) else [v] for v in obs.values()]
        flattened_obs = [item for sublist in flattened_obs for item in sublist]
        return flattened_obs
