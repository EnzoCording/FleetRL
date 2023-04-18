import pandas as pd
from pandas import Timestamp

from FleetRL.utils.time_picker.time_picker_base import TimePickerBase


class StaticTimePicker(TimePickerBase):
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        chosen_start_time = pd.to_datetime("1/1/2020 10:00")

        # return start time
        return chosen_start_time
