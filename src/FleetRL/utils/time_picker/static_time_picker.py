import pandas as pd
from pandas import Timestamp

from FleetRL.utils.time_picker.time_picker import TimePicker


class StaticTimePicker(TimePicker):

    def __init__(self, start_time: str = "04/05/2020 15:00"):
        self.start_time = start_time

    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        chosen_start_time = pd.to_datetime(self.start_time)
        # return start time
        return chosen_start_time
