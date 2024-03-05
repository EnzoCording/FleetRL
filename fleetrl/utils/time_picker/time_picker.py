import pandas as pd
from pandas import Timestamp

class TimePicker:
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        """
        Parent class for time picker objects
        :param db: dataframe from env
        :param freq: frequency specification string for pandas
        :param end_cutoff: amount of days cut off at the end to allow some buffer. In the eval time picker case,
        the end cutoff specifies the size of the validation set.
        :return: A chosen time stamp
        """
        raise NotImplementedError("This is an abstract class.")
