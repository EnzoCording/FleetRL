import numpy as np
import pandas as pd

from FleetRL.utils.observation.observer import Observer


class ObserverWithBuildingLoad(Observer):
    def get_obs(self, db: pd.DataFrame, price_lookahead: int, time: pd.Timestamp) -> list:

        soc = db.loc[(db['date'] == time), 'SOC_on_return'].values
        hours_left = db.loc[(db['date'] == time), 'time_left'].values

        price_start = np.where(db["date"] == time)[0][0]
        price_end = np.where(db["date"] == (time + np.timedelta64(price_lookahead, 'h')))[0][0]
        price = db["DELU"][price_start: price_end].values

        building_load = db.loc[db["date"] == time, "load"].values

        return [soc, hours_left, price, building_load]
