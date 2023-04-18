import pandas as pd
from pandas import Timestamp

class TimePickerBase():
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        raise NotImplementedError("This is an abstract class.")