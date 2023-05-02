import numpy as np
import pandas as pd


class Observer:
    def get_obs(self, db: pd.DataFrame, price_window_size: int,
                time: pd.Timestamp) -> list:
        """
        :param db: from the env
        :param price_window_size: from the env
        :param time: from the env
        :return: Returns a list of np arrays that make up different parts of the observation.
        """
        raise NotImplementedError("This is an abstract class")