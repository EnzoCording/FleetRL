import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer


class BasicObserver(Observer):
    def get_obs(self, db: pd.DataFrame, spot_price: pd.DataFrame, price_lookahead: int,
                time: pd.Timestamp) -> list:

        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        price_start = np.where(spot_price["date"] == time)[0][0]
        price_end = np.where(spot_price["date"] == (time + np.timedelta64(price_lookahead, 'h')))[0][0]
        price = spot_price["DELU"][price_start: price_end].values

        return [soc, hours_left, price]
