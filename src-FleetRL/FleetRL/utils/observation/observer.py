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

    # Always the same, so can be defined in base class
    @staticmethod
    def get_trip_len(db: pd.DataFrame, car: int, time: pd.Timestamp) -> float:
        """
        :param db: from the env
        :param car: car ID
        :param time: current timestamp
        :return: length of trip in hours as a float
        """
        raise NotImplementedError("This is an abstract class")
