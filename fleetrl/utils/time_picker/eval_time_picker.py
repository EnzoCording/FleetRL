import random

import numpy as np
import pandas as pd
from pandas import Timestamp
from fleetrl.utils.time_picker.time_picker import TimePicker

class EvalTimePicker(TimePicker):

    """
    Time picker for validation set. The amount of days and thus the train/validation split is set in time config.

    """

    def __init__(self, ep_len):
        self.episode_length = ep_len


    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:

        """
        Randomly chooses a start time from the validation set.

        :param db: Database
        :param freq: Time frequency
        :param end_cutoff: This is the size of the validation window. By default, the end_cutoff is 2 months, so Nov
            and Dec are the validation set.
        :return: start time, pd.TimeStamp

        """

        # possible start times: remove last X days based on end_cutoff
        possible_start_times = pd.date_range(start=(db["date"].max() - np.timedelta64(end_cutoff, 'D')),
                                             end=(db["date"].max() - np.timedelta64(2 * self.episode_length, 'h')),
                                             freq=freq
                                             )

        # choose a random start time and start the episode there
        chosen_start_time = random.choice(possible_start_times)

        first_year = db.iloc[0]["date"].year
        last_year = db.iloc[-1]["date"].year
        chosen_year = chosen_start_time.year

        # keep month, day and time but set the right year to match with schedule database
        if (chosen_year < first_year) or (chosen_year > last_year):
            print(f"Chosen start year: {chosen_year}, Start year in database: {first_year}")
            print("Chosen year does not match db years. Adjusting to match start year in db...")
            chosen_start_time = chosen_start_time + pd.DateOffset(years=first_year-chosen_year)

        # return start time
        return chosen_start_time
