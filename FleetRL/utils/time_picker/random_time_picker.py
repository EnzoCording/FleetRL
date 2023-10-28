import random

import numpy as np
import pandas as pd
from pandas import Timestamp
from FleetRL.utils.time_picker.time_picker import TimePicker


class RandomTimePicker(TimePicker):
    """
    Picks a random time from the training set.
    """
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:

        """
        Randomly chooses a start time from the validation set.

        :param db: Database
        :param freq: Time frequency
        :param end_cutoff: Buffer that avoids problem with lookaheads
        :return: start time, pd.TimeStamp
        """

        # possible start times: remove last X days based on end_cutoff
        possible_start_times = pd.date_range(start=db["date"].min(),
                                             end=(db["date"].max() - np.timedelta64(end_cutoff, 'D')),
                                             freq=freq
                                             )

        # choose a random start time and start the episode there
        chosen_start_time = random.choice(possible_start_times)

        # return start time
        return chosen_start_time
