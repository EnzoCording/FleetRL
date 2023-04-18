import random

import numpy as np
import pandas as pd
from pandas import Timestamp
from FleetRL.utils.time_picker.time_picker import TimePicker


class RandomTimePicker(TimePicker):
    def choose_time(self, db: pd.Series, freq: str, end_cutoff: int) -> Timestamp:
        # possible start times: remove last X days based on end_cutoff
        # TODO: same day can be pulled multiple times - is this a problem?
        possible_start_times = pd.date_range(start=db["date"].min(),
                                             end=(db["date"].max() - np.timedelta64(end_cutoff, 'D')),
                                             freq=freq
                                             )

        # choose a random start time and start the episode there
        chosen_start_time = random.choice(possible_start_times)

        # return start time
        return chosen_start_time
