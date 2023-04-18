import numpy as np
import pandas as pd


class Observer:
    def get_obs(self, db: pd.DataFrame, spot_price: pd.DataFrame, price_window_size: int,
                time: pd.Timestamp) -> list:
        raise NotImplementedError("This is an abstract class")