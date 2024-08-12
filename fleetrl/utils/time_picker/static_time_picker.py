import pandas as pd
from pandas import Timestamp

from fleetrl.utils.time_picker.time_picker import TimePicker


class StaticTimePicker(TimePicker):
    """
    Picks a static / always the same starting time.
    """

    def __init__(self, start_time: str = "01/02/2021 19:00"):
        """
        :param start_time: When initialised, start time is specified
        """
        self.start_time = start_time

    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:

        first_year = db.iloc[0]["date"].year
        last_year = db.iloc[-1]["date"].year
        chosen_start_time = pd.to_datetime(self.start_time)
        chosen_year = chosen_start_time.year

        # keep month, day and time but set the right year to match with schedule database
        if (chosen_year < first_year) or (chosen_year > last_year):
            print(f"Chosen start year: {chosen_year}, Start year in database: {first_year}")
            print("Chosen year does not match db years. Adjusting to match start year in db...")
            chosen_start_time = chosen_start_time + pd.DateOffset(years=first_year-chosen_year)

        # return start time
        return chosen_start_time
