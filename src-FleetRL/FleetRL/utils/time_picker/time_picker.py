import pandas as pd
from pandas import Timestamp


class TimePicker:
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        raise NotImplementedError("This is an abstract class.")
