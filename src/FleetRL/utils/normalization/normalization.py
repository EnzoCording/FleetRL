import numpy as np


class Normalization:
    """
    Parent class for Normalization.
    """
    def normalize_obs(self, obs: dict) -> np.ndarray:
        """
        Normalizes the values in an observation.

        :param obs: An observation, containing the state of charge, hours left, and price
        :return: The normalized observation, concatenated to a single array
        """
        raise NotImplementedError("This is an abstract class")

    def make_boundaries(self, dim: tuple[int]) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Create the observation space for gym. Depending on the observation, upper and lower boundaries can change.

        :param dim: The length of the array, depends on the number of cars, and other information contained in the obs
        :return: A tuple, containing the low obs and high obs array that will be parsed to gym.Spaces.box
        """
        raise NotImplementedError("This is an abstract class")
