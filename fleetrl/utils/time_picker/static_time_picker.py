import pandas as pd
from pandas import Timestamp

from fleetrl.utils.time_picker.time_picker import TimePicker


class StaticTimePicker(TimePicker):
    """
    Picks a static / always the same starting time.
    """

    def __init__(self, start_time: str = "01/01/2020 00:00"):
        """
        :param start_time: When initialised, start time is specified
        """
        self.start_time = start_time

    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        chosen_start_time = pd.to_datetime(self.start_time)
        # return start time
        return chosen_start_time
